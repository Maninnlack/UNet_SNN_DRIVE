import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, surrogate

class DoubleConv_INN(nn.Module):
    def __init__(self, in_channels, out_channels, v_threshold=1.0, v_reset=0.0):
        super(DoubleConv_INN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        return self.conv(x)

class UNet_INN(nn.Module):
    def __init__(self, in_channels, out_channels, T=5, v_threshold=1.0, v_reset=0.0):
        super(UNet_INN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T = T
        
        self.conv1 = DoubleConv_INN(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv_INN(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv_INN(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv_INN(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv_INN(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv_INN(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv_INN(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv_INN(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv_INN(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # 输出形式是否正确？
        out_spike_counter = c10
        
        # for _ in range(self.T):
        #     out_spike_counter += self.conv10(c9)
        # return out_spike_counter / self.T
        return out_spike_counter
