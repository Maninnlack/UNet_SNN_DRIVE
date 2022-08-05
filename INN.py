import argparse
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils.distributed_utils as dis_utils
from dataset import DriveDataset, get_transform
from Unet_INN import UNet_INN
from utils.dice_coefficient_loss import build_target, dice_loss
from utils.loss import DiceLoss, FocalLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def criterion(inputs, target, loss_weight=None, num_classes:int = 2, dice: bool = True, ignore_index: int = -100):
    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
    return loss


def main():
    # image and mask path
    data_root = './data'

    checkpoint_path = './model_save'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    parse = argparse.ArgumentParser(description='UNet-INN')
    parse.add_argument('--device', default='cuda:0', help='运行的设备')
    parse.add_argument('-b', '--batchsize', default=2, type=int, help='Batch 大小')
    parse.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parse.add_argument('-e', '--epoch', default=200, type=int, help='epoch数量')
    parse.add_argument('-T', default=5, type=int)
    # parse.add_argument('-L', '--loss', default='F', type=str, help='损失函数')

    arg = parse.parse_args()

    #  nvidia configure
    device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # hyper perameters
    batch_size = arg.batchsize
    epochs = arg.epoch
    learning_rate = arg.learning_rate
    num_classes = 2
    T = arg.T
    # 用来保存训练以及验证过程中信息
    results_file = "results.txt"

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # dataset
    train_dataset = DriveDataset(data_root, 
                                 dtype='train', 
                                 transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = DriveDataset(data_root,
                               dtype='val', 
                               transforms=get_transform(train=True, mean=mean, std=std))

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size,
                                pin_memory=True,
                                collate_fn=train_dataset.collate_fn)

    # net init
    model_name = 'INN_UNet_Drive'
    net = UNet_INN(in_channels=3, out_channels=num_classes)
    net.to(device=device)

    # optimizer, loss function, learning rate
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # tensorboard 可视化
    writer = SummaryWriter(comment=model_name)
    best_dice = 0.
    start_time = time.time()
    for epoch in range(epochs):
        mean_loss = train_one_epoch(net, optimizer, train_dataloader, device, epoch, T, num_classes, print_freq=10)
        confmat, dice = evaluate(net, val_dataloader, device, num_classes, T)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        writer.add_scalar('train loss', mean_loss, epoch)
        writer.add_scalar('val dice', dice, epoch)

        # write to txt 
        with open(results_file, "a") as f:
            train_info = f"train loss: {mean_loss:.4f}  dice coefficient: {dice:.3f} [epoch: {epoch}/{epochs}]\n"
            f.write(train_info + val_info + "\n\n")

        if best_dice < dice:
            best_dice = dice
            torch.save(net.state_dict(), checkpoint_path + os.sep + model_name + '.pth')
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=(total_time)))
    print("training time {}".format(total_time_str))
    writer.close()


def compute_IoU(pred, mask):
    pred_mask = torch.zeros_like(pred)
    pred_mask[pred > 0.4] = 1.0

    epsilon=1e-6
    inter = torch.dot(pred_mask.reshape(-1), mask.reshape(-1))
    set_sum = torch.sum(pred_mask) + torch.sum(mask)
    if set_sum == 0:
        set_sum = 2 * inter
    return (2 * inter + epsilon) / (set_sum + epsilon)


def train_one_epoch(model, optimizer, data_loader, device, epoch, T, num_classes, print_freq=10):
    model.train()
    metric_logger = dis_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dis_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None
    
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        for t in range(T):
            if t == 0:
                output = model(image)
            else:
                output += model(image)
        output /= T

        loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        functional.reset_net(model)

        metric_logger.update(loss=loss.item(), lr=1e-3)

    return metric_logger.meters["loss"].global_avg


def evaluate(model, data_loader, device, num_classes, T):
    model.eval()
    confmat = dis_utils.ConfusionMatrix(num_classes)
    dice = dis_utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = dis_utils.MetricLogger(delimiter="  ")
    header = 'Test: '
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            for t in range(T):
                if t == 0:
                    output = model(image)
                else:
                    output += model(image)
            output /= T

            functional.reset_net(model)

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)
        
        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
    
    return confmat, dice.value.item()



if __name__ == '__main__':
    main()
