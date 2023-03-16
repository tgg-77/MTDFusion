import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import fusion_strategy


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.gelu(out)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*3, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*4, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# DenseFuse network
class TDFusion_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(TDFusion_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 48, 80, 96]
        kernel_size = 3
        stride = 1

        # extract
        self.conv_0_1 = ConvLayer(1, nb_filter[0], kernel_size, stride)
        self.conv_0_2 = ConvLayer(nb_filter[0], nb_filter[1], kernel_size, stride)

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)



        # decoder
        self.conv1_1 = ConvLayer(nb_filter[5], nb_filter[5], kernel_size, stride)
        self.conv2 = ConvLayer(nb_filter[5], nb_filter[4], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[4], nb_filter[1], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[1], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], nb_filter[2], kernel_size, stride)
        self.conv6 = ConvLayer(nb_filter[2], nb_filter[0], kernel_size, stride)
        self.conv7 = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)

    def encoder(self, input):
        # x0_1 = self.conv_0_1(input)
        # x0_2 = self.conv_0_2(x0_1)
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        return [x_DB]

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     # addition
    #     if strategy_type is 'attention_weight':
    #         # attention weight
    #         fusion_function = fusion_strategy.attention_fusion_weight
    #     else:
    #         fusion_function = fusion_strategy.addition_fusion
    #
    #     f_0 = fusion_function(en1[0], en2[0])
    #     return [f_0]

    def fusion(self, en1, en2, strategy_type='addition'):
        f_0 = (en1[0] + en2[0])/2
        return [f_0]

    def fusion_3(self, en1, en2, en3, strategy_type='addition'):
        f_0 = (en1[0] + en2[0] + en3[0])/3
        return [f_0]

    def fusion_1(self, en1, en2, en3, strategy_type='addition'):
        f_0 = en1[0]*0.4 + en2[0]*0.4 + 0.2*en3[0]
        return [f_0]

    def fusion_max(self, en1, en2, en3, strategy_type='addition'):
        f_0 = torch.max(en1[0], en3[0])
        f_0 = torch.max(f_0[0], en2[0])
        return [f_0]

    def decoder(self, f_en):
        x2_2 = self.conv1_1(f_en[0])
        x2 = self.conv2(x2_2)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        output = self.conv7(x6)

        return [output]


def test():
    print('start test')
    x = torch.randn((1, 1, 482, 512))
    # x = torch.randn((1, 3, 256, 256))
    model = TDFusion_net(input_nc=1, output_nc=1)
    en = model.encoder(x)
    print(en[0].shape)
    predictions = model.decoder(en)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(predictions[0].shape)


if __name__ == "__main__":
    test()