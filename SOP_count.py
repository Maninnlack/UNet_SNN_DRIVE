import torch
from torchvision import transforms
from collections import Counter
import numpy as np
from PIL import Image

from Unet_INN import UNet_INN
from Unet_ANN import UNet
from spikingjelly.activation_based import monitor, functional, neuron


def SOP_count(weight_path, img_path, roi_mask_path, device):
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # create model
    T = 5
    model = UNet_INN(3, 2)

    # model load weights
    model.load_state_dict(torch.load(weight_path))
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert("L")
    t = transforms.Resize((256, 256))
    roi_img = np.array(t(roi_img))

    # load img
    original_img = Image.open(img_path).convert("RGB")

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=mean, std=std)
    ])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = 256, 256
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)

        for t in range(T):
            model(init_img)
            functional.reset_net(model)

        spike_seq_monitor = monitor.OutputMonitor(model, neuron.IFNode)
        for t in range(T):
            if t == 0:
                output = model(img.to(device))
            else:
                output += model(img.to(device))
        output /= T
        print(f'len of spike_seq_monitor_monitor.records=\n{len(spike_seq_monitor.records)}')

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
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    weights_path = './model_save/INN_UNet_Drive.pth'
    img_path = './data/test/images/16_test.tif'
    roi_mask_path = './data/test/mask/16_test_mask.gif'
    SOP_count(weights_path, img_path, roi_mask_path, device)
