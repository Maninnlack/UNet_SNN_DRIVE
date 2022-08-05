import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional, monitor, neuron
from torch.utils.data import DataLoader

from dataset import ImageDataset
from Unet_INN import UNet_INN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')
print(f'Using {device} device.')

test_img_path = './data_cut/test1'
test_mask_path = './data_cut/test1_mask'

model_path = './model_save/INN_UNet_data_cut_T_max_F.pth'

def test_func(test_img_path, test_mask_path, model_path, T=5):
    predicted_img = []
    test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = UNet_INN(3, 1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for _, batch in enumerate(test_dataloader):
        img = batch['img']
        mask = batch['mask']
        img_g = img.to(device)

        with torch.no_grad():
            for t in range(T):
                if t == 0:
                    pred = model(img_g)
                else:
                    pred += model(img_g)
            pred /= T
            predicted_img.append(pred)
    pred_show_img = predicted_img[0].squeeze()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title('image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('label')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow((torch.sigmoid(pred_show_img)).float().cpu().numpy(), cmap='gray')
    plt.title('INN segmentation')
    plt.axis('off')
    plt.savefig('./test_INN_cut_max_D.jpg', dpi=800)
    plt.show()

def test_SOP(test_img_path, test_mask_path, model_path, T=5):

    test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = UNet_INN(3, 1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    spike_seq_monitor = monitor.OutputMonitor(model, neuron.IFNode)

    for _, batch in enumerate(test_dataloader):
        img = batch['img']
        mask = batch['mask']
        img_g = img.to(device)

        with torch.no_grad():
            for t in range(T):
                if t == 0:
                    pred = model(img_g)
                    print(f'len of spike_seq_monitor_monitor.records=\n{len(spike_seq_monitor.records)}')
                else:
                    pred += model(img_g)
                    # print(f'len of spike_seq_monitor_monitor.records=\n{len(spike_seq_monitor.records)}')
            pred /= T
        print(f'len of spike_seq_monitor_monitor.records=\n{len(spike_seq_monitor.records)}')
        functional.reset_net(model)

    SOP_list = []
    for i in range(len(spike_seq_monitor.records)):
        SOP_list.append(Counter(spike_seq_monitor.records[i].cpu().numpy().reshape(-1)))
    SOP_list_spike = [a[1.0] for a in SOP_list]
    print(f'SOP of INN on test :{sum(SOP_list_spike)}')
    with open('./SOP_result.txt', 'w+') as f:
        for i in range(len(SOP_list_spike)):
            f.write(f'{SOP_list_spike[i]}\n')
        f.write(f'SOP of INN on test :{sum(SOP_list_spike)}')
    

if __name__ == '__main__':
    test_func(test_img_path, test_mask_path, model_path)
    test_SOP(test_img_path, test_mask_path, model_path)
