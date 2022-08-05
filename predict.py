import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from Unet_INN import UNet_INN
from Unet_ANN import UNet
from spikingjelly.activation_based import functional


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def test_plot_img(weights_path, img_path, roi_mask_path, device, model_type='INN'):

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # create model
    if model_type == 'INN':
        T = 5
        model = UNet_INN(3, 2)
    else:
        model = UNet(3, 2)

    # model load weights
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    t = transforms.Resize((256, 256))
    roi_img = np.array(t(roi_img))

    # load img
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((256, 256)),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # evaluation
    with torch.no_grad():
        # init model
        img_height, img_width = 256, 256
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)

        if model_type == 'INN':
            for t in range(T):
                model(init_img)
                functional.reset_net(model)
        else:
            model(init_img)

        t_start = time_synchronized()
        if model_type == 'INN':
            for t in range(T):
                if t == 0:
                    output = model(img.to(device))
                else:
                    output += model(img.to(device))
            output /= T
        else:
            output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time:{}".format(t_end - t_start))

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # foreground = 255 background = 0
        prediction[prediction == 1] = 255
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("{}_test.png".format(model_type))


def main():
    classes = 1
    model_type = 'ANN'
    weights_path = './model_save/{}_UNet_Drive.pth'.format(model_type)
    img_path = './data/test/images/16_test.tif'
    roi_mask_path = './data/test/mask/16_test_mask.gif'
    assert os.path.exists(weights_path), f"weights {weights_path} does not exists."
    assert os.path.exists(img_path), f"weights {img_path} does not exists."
    assert os.path.exists(roi_mask_path), f"weights {roi_mask_path} does not exists."

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    test_plot_img(weights_path, img_path, roi_mask_path, device, model_type=model_type)


if __name__ == '__main__':
    main()