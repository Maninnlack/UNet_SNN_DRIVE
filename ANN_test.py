import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import ImageDataset
from Unet_ANN import UNet

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')
print(f'Using {device} device.')

test_img_path = './data/test1'
test_mask_path = './data/test1_mask'
# test_img_path = './data_test/test'
# test_mask_path = './data_test/test'

model_path = './model_save/ANN_UNet_data.pth'


def test_func(test_img_path, test_mask_path, model_path):
    prediced_img = []
    test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    model = UNet(3,1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for index, batch in enumerate(test_dataloader):
        img = batch['img']
        mask = batch['mask']
        img_g = img.to(device)

        with torch.no_grad():
            pred = model(img_g)
            prediced_img.append(pred)
    pred_show_img = prediced_img[0].squeeze()
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img.squeeze(0).permute(1,2,0).cpu())
    plt.axis('off')
    plt.title('image')
    plt.subplot(1,3,2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('label')
    plt.subplot(1,3,3)
    plt.imshow((pred_show_img > 0.4).float().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('ANN segmentation')
    plt.savefig('./test_ANN.jpg')
    plt.show()


if __name__ == '__main__':
    test_func(test_img_path, test_mask_path, model_path)