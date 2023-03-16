import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi) # 原来的代码有问题，写成了conv_ir，现在重新训练
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out


class Fusion_network(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type
        self.conv_ir_1 = ConvLayer(1, 64, 3, 1)
        self.conv_vis_1 = ConvLayer(1, 64, 3, 1)
        self.conv_d_1 = ConvLayer(1, 64, 3, 1)


        self.fusion_block1 = FusionBlock_res(nC[0], 0)

        self.conv_fusion_1 = ConvLayer(128, 32, 3, 1)
        self.conv_fusion_2 = ConvLayer(32, 1, 3, 1)

    def forward(self, en_ir, en_vi, en_d):
        en_ir = self.conv_ir_1(en_ir)
        en_vi = self.conv_vis_1(en_vi)
        en_d = self.conv_d_1(en_d)
        f1_0 = self.fusion_block1(en_ir, en_vi)
        f1_1 = torch.cat([f1_0, en_d], 1)
        f = self.conv_fusion_1(f1_1)
        f = self.conv_fusion_2(f)

        return f


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
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


def test():
    print('start test')
    x = torch.randn((1, 1, 482, 512))
    # x = torch.randn((1, 3, 256, 256))
    model = Fusion_network([64], 'res')
    predictions = model(x, x, x)
    print(predictions.shape)


if __name__ == "__main__":
    test()