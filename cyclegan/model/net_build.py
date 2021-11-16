import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.func=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features))
    def forward(self,x):
        x1=self.func(x)
        return x+x1

class convNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.func=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.func(x)
        return x

class convTNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.func=nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding ,output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.func(x)
        return x

class generator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.func0 = nn.ReflectionPad2d(3)
        self.func1 = convNormRelu(n_channels, 64, 7,stride=1 ,padding=0)
        self.func2_1=convNormRelu(64,128)
        self.func2_2=convNormRelu(128,256)
        self.block = nn.Sequential(*[ResidualBlock(256) for i in range(9)])
        self.func3_1 = convTNormRelu(256,128)
        self.func3_2 = convTNormRelu(128,64)
        self.func4 = nn.Sequential( nn.ReflectionPad2d(3),
                                    nn.Conv2d(64, n_classes, 7),
                                    nn.Tanh())

    def forward(self, x):
        x1 = self.func0(x)
        x2 = self.func1(x1)
        x3 = self.func2_2(self.func2_1(x2))
        x3 = self.block(x3)
        x4 = self.func3_2(self.func3_1(x3))

        return self.func4(x4)

class discriminator(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.func1 = nn.Sequential( nn.Conv2d(n_channels, 64, 4, stride=2, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(64, 128, 4, stride=2, padding=1),
                                    nn.InstanceNorm2d(128), 
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(128, 256, 4, stride=2, padding=1),
                                    nn.InstanceNorm2d(256), 
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(256, 512, 4, padding=1),
                                    nn.InstanceNorm2d(512), 
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(512, 1, 4, padding=1))
    def forward(self, x):
        x1 = self.func1(x)
        return F.avg_pool2d(x1, x1.size()[2:]).view(x1.size()[0], -1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)