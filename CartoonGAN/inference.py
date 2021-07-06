import cv2
import torch
import numpy as np
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        """
        When run the model, the input will be passed to this function
        :param inputs: Tensor converted from image
        :return: Tensor
        """
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)

        return output


class DownBlock(nn.Module):
    """
    Down sampling class
    """

    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """
        When run the model, the input will be passed to this function
        :param inputs: Tensor
        :return: Down sampled tensor
        """
        output = self.conv_layer(inputs)
        return output


class UpBlock(nn.Module):
    """
    Up sampling class
    """

    def __init__(self, in_channel, out_channel, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        )
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.last_act = nn.Tanh()

    def forward(self, inputs):
        """
        When run the model, the input will be passed to this function
        :param inputs: Tensor
        :return: Up sampled tensor
        """
        output = self.conv_layer(inputs)
        return self.last_act(output) if self.is_last else self.act(output)


class Generator(nn.Module):
    def __init__(self, num_channel=32, num_blocks=4):
        super(Generator, self).__init__()

        self.down1 = DownBlock(3, num_channel)
        self.down2 = DownBlock(num_channel, num_channel * 2)
        self.down3 = DownBlock(num_channel * 2, num_channel * 3)
        self.down4 = DownBlock(num_channel * 3, num_channel * 4)

        res_blocks = [ResBlock(num_channel * 4)] * num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)

        self.up1 = UpBlock(num_channel * 4, num_channel * 3)
        self.up2 = UpBlock(num_channel * 3, num_channel * 2)
        self.up3 = UpBlock(num_channel * 2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)

    def forward(self, input):
        # Down samplings
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        down4 = self.res_blocks(down4)

        # Up samplings
        up1 = self.up1(down4)
        up2 = self.up2(up1 + down3)
        up3 = self.up3(up2 + down2)
        up4 = self.up4(up3 + down1)

        return up4


if __name__ == '__main__':
    weight = torch.load('weight.pth', map_location='cpu')
    model = Generator()
    model.load_state_dict(weight)
    model.eval()

    raw_image = cv2.imread('./white_box/images/00054.jpg')
    # raw_image = cv2.resize(raw_image, (256, 256), interpolation=cv2.INTER_AREA)
    image = raw_image / 127.5 - 1

    cv2.imshow('test', image)
    cv2.waitKey(0)

    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)

    output = model(image.float())
    output = output.squeeze(0).detach().numpy()
    output = output.transpose(1, 2, 0)
    output = (output + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    cv2.imshow('Cartooned Image', output)
    cv2.waitKey(0)
