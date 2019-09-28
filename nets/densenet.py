import torch
from torch import nn
import torch.nn.init as init
# from groupnorm import GroupNorm


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # GroupNorm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # GroupNorm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            # GroupNorm(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

    def forward(self, x):
        return (self.conv(x), self.conv(x))  # (N, 128, 75, 75)


class DenseLayer(nn.Module):
    def __init__(self, in_channel, mid_channel=129, k=48):
        super(DenseLayer, self).__init__()
        self.model_name = 'DenseLayer'

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            # GroupNorm(in_channel, num_groups=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            # GroupNorm(mid_channel, num_groups=43),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, k, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        _x = self.conv(x)
        x = torch.cat([x, _x], dim=1)
        return x


class DenseBlock(nn.Module):
    def __init__(self, layer_num, in_channel, mid_channel=129, k=48):
        super(DenseBlock, self).__init__()
        self.model_name = 'DenseBlock'

        layers = []
        layers.append(DenseLayer(in_channel, mid_channel))

        for idx in range(1, layer_num):
            layers.append(DenseLayer(in_channel+k*idx, mid_channel))

        self.dense = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channel, out_channel, pool=False):
        super(TransitionLayer, self).__init__()
        self.model_name = 'TransitionLayer'
        self.is_pool = pool

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            # GroupNorm(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_pool:
            return (x, self.pool(x))
        else:
            return (x, x)


class DenseSupervision1(nn.Module):
    def __init__(self, in_channel, out_channel=256):
        super(DenseSupervision1, self).__init__()
        self.model_name = 'DenseSupervision'

        self.right = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.BatchNorm2d(in_channel),
            # GroupNorm(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, bias=False)
        )

    def forward(self, x1, x2):
        # x1 should be f1
        right = self.right(x1)
        return torch.cat([x2, right], 1)


class DenseSupervision(nn.Module):
    def __init__(self, in_channel, out_channel=128):
        super(DenseSupervision, self).__init__()
        self.model_name = 'DenseSupervision'

        self.left = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.BatchNorm2d(in_channel),
            # GroupNorm(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, bias=False)
        )
        self.right = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            # GroupNorm(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            # GroupNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 2, 1, bias=False)
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return torch.cat([left, right], 1)


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.model_name = 'DenseNet'

        self.stem = Stem()
        self.dense1 = DenseBlock(6, 128)
        self.trans1 = TransitionLayer(416, 416, pool=True)
        self.dense2 = DenseBlock(8, 416)
        self.trans2 = TransitionLayer(800, 800, pool=True)
        self.dense3 = DenseBlock(8, 800)
        self.trans3 = TransitionLayer(1184,
                                      1184)  # TODO: it is 1120 in the paper, but 1184 in the code, and this matchs the growthrate of the next densenetblock
        self.dense4 = DenseBlock(8, 1184)
        # TODO: output size in the paper is 1568
        self.trans4 = TransitionLayer(1568, 256)

        self.dense_sup1 = DenseSupervision1(800, 256)
        self.dense_sup2 = DenseSupervision(512, 256)
        self.dense_sup3 = DenseSupervision(512, 128)
        self.dense_sup4 = DenseSupervision(256, 128)
        self.dense_sup5 = DenseSupervision(256, 128)

    def forward(self, x):
        lower_x, x = self.stem(x)  # n, 128, 128, 128

        x = self.dense1(x)  # n, 416, 128, 128
        _, x = self.trans1(x)  # 2, 416, 64, 64

        x = self.dense2(x)
        f1, x = self.trans2(x)

        x = self.dense3(x)
        _, x = self.trans3(x)

        x = self.dense4(x)
        _, x = self.trans4(x)

        f2 = self.dense_sup1(f1, x)
        f3 = self.dense_sup2(f2)
        f4 = self.dense_sup3(f3)
        f5 = self.dense_sup4(f4)
        f6 = self.dense_sup5(f5)
        return f1, f2, f3, f4, f5, f6, lower_x

def test():
    densenet = DenseNet()
    input = torch.randn(2, 3, 512, 512)
    output = densenet(input)

    for i in output:
        print(i.shape)

if __name__ == '__main__':
    test()
