import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self,c_in, c_out, k, s, p, bias=False):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in,c_out,k,s,p,bias),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self,x):
        return self.conv(x)

class ConvResidual(nn.Module):
    def __init__(self,c_in):
        super(ConvResidual, self).__init__()
        c = c_in // 2
        self.conv = nn.Sequential(
            Convolution(c_in,c,1,1,0),
            Convolution(c,c_in,3,1,1)
        )

    def forward(self,x):
        return x + self.conv(x)

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = Convolution(3,32,3,1,1)
        self.conv2 = Convolution(32,64,3,2,1)
        self.conv3_4 = ConvResidual(64)
        self.conv5 = Convolution(64,128,3,2,1)
        self.conv6_9 = nn.Sequential(
            ConvResidual(128),
            ConvResidual(128)
        )
        self.conv10 = Convolution(128,256,3,2,1)
        self.conv11_26 = nn.Sequential(
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256)
        )
        self.conv27 = Convolution(256,512,3,2,1)
        self.conv28_43 = nn.Sequential(
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512)
        )
        self.conv44 = Convolution(512,1024,3,2,1)
        self.conv45_52 = nn.Sequential(
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024)
        )

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3_4 = self.conv3_4(conv2)
        conv5 = self.conv5(conv3_4)
        conv6_9 = self.conv6_9(conv5)
        conv10 = self.conv10(conv6_9)
        conv11_26 = self.conv11_26(conv10)
        conv27 = self.conv27(conv11_26)
        conv28_43 = self.conv28_43(conv27)
        conv44 = self.conv44(conv28_43)
        conv45_52 = self.conv45_52(conv44)
        return conv45_52, conv28_43, conv11_26





