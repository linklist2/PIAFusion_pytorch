import math

from torch import nn

from models.common import reflect_conv


class Illumination_classifier(nn.Module):
    def __init__(self, input_channels, init_weights=True):
        super(Illumination_classifier, self).__init__()
        self.conv1 = reflect_conv(in_channels=input_channels, out_channels=16)
        self.conv2 = reflect_conv(in_channels=16, out_channels=32)
        self.conv3 = reflect_conv(in_channels=32, out_channels=64)
        self.conv4 = reflect_conv(in_channels=64, out_channels=128)

        self.linear1 = nn.Linear(in_features=128, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化权重

        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        activate = nn.LeakyReLU(inplace=True)
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        # x = activate(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)  # 设置ReLU激活函数，过滤负值
        # x = nn.Sigmoid()(x)
        # x = nn.ReLU(inplace=True)(x)
        return x
