import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import DriveDataset
from Unet_ANN import UNet
from Unet_INN import UNet_INN
from spikingjelly.activation_based import functional

def test_plot_func(test_img_path, test_mask_path, model_path_ANN, model_path_INN, device, T=5):
    prediced_img = []
    test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    model_ANN = UNet(3,1).to(device)
    model_INN = UNet_INN(3,1).to(device)

    model_ANN.load_state_dict(torch.load(model_path_ANN))
    model_INN.load_state_dict(torch.load(model_path_INN))

    model_ANN.eval()
    model_INN.eval()

    for _, batch in enumerate(test_dataloader):
        img = batch['img']
        mask = batch['mask']
        img_gpu = img.to(device)

        with torch.no_grad():
            pred_ANN = model_ANN(img_gpu)
            for t in range(T):
                if t == 0:
                    pred_INN = model_INN(img_gpu)
                else:
                    pred_INN += model_INN(img_gpu)
            pred_INN /= T
            functional.reset_net(model_INN)

            prediced_img.append(pred_ANN)
            prediced_img.append(pred_INN)

    pred_show_img_ANN = prediced_img[0].squeeze()
    pred_show_img_INN = prediced_img[1].squeeze()
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(img.squeeze(0).permute(1,2,0).cpu())
    plt.axis('off')
    plt.title('image')
    plt.subplot(1,4,2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('mask')
    plt.subplot(1,4,3)
    plt.imshow((pred_show_img_ANN > 0.5).float().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('ANN result')
    plt.subplot(1,4,4)
    plt.imshow((torch.sigmoid(pred_show_img_INN ) > 0.5).float().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('INN result')
    plt.savefig('./test_CT_T_max_half.jpg', dpi=800)
    plt.show()

def compute_IoU(pred, mask):
    pred_mask = torch.zeros_like(pred)
    pred_mask[pred > 0.4] = 1.0

    epsilon=1e-6
    inter = torch.dot(pred_mask.reshape(-1), mask.reshape(-1))
    set_sum = torch.sum(pred_mask) + torch.sum(mask)
    if set_sum == 0:
        set_sum = 2 * inter
    return (2 * inter + epsilon) / (set_sum + epsilon)


def test_IoU(test_img_path, test_mask_path, model_path, device, dtype='ANN', bs=2, T=5):
    test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=bs)    # 因为同时运行两个网络，因此batchsize要减半
    
    if dtype == 'ANN':
        model_ANN = UNet(3,1).to(device)
        model_ANN.load_state_dict(torch.load(model_path))
        model_ANN.eval()
    else:
        model_INN = UNet_INN(3,1).to(device)
        model_INN.load_state_dict(torch.load(model_path_INN))
        model_INN.eval()

    IoU_list = []

    for _, batch in enumerate(test_dataloader):
        img = batch['img']
        mask = batch['mask']
        img_gpu = img.to(device)
        mask_gpu = mask.to(device)

        if dtype == 'ANN':
            pred = model_ANN(img_gpu)
        else: 
            for t in range(T):
                if t == 0:
                    pred = model_INN(img_gpu)
                else:
                    pred += model_INN(img_gpu)
            pred /= T
            functional.reset_net(model_INN)

        IoU = compute_IoU(pred, mask_gpu)

        IoU_list.append(IoU)

    return sum(IoU_list)/len(IoU_list)


if __name__ == '__main__':


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(f'Using {device} device.')

    test_img_path = './data_cut/test'
    test_mask_path = './data_cut/test_mask'

    model_path_ANN = './model_save/ANN_UNet_data.pth'
    model_path_INN = './model_save/INN_UNet_data_cut_T_max_F.pth'

    # IoU_ANN = test_IoU(test_img_path, test_mask_path, model_path_ANN, device, dtype='ANN', bs=8)
    # IoU_INN = test_IoU(test_img_path, test_mask_path, model_path_INN, device, dtype='INN', bs=2)
    # print('{} ANN IoU:{} \n{} INN IoU:{}'.format(model_path_ANN.split('/')[-1][:-4], IoU_ANN, model_path_INN.split('/')[-1][:-4] ,IoU_INN))


    test_img_path = './data_cut/test1'
    test_mask_path = './data_cut/test1_mask'
    test_plot_func(test_img_path, test_mask_path, model_path_ANN, model_path_INN, device=device)