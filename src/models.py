import numpy as np
import torch
from torch import nn

class conv_block(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel,out_channel,3,1,1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(out_channel,out_channel,3,1,1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
    def forward(self,input):
        return self.conv(input)

class up_conv(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super(up_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channel,out_channel,3,1,1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self,input):
        return self.conv(input)

class NonLocal1D(nn.Module):
    def __init__(self, in_channels) -> None:
        super(NonLocal1D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels//2 # bottleneck design (see section 3.3)

        self.theta = nn.Conv1d(self.in_channels, self.inter_channels, 1)
        self.phi = nn.Sequential(
            nn.Conv1d(self.in_channels, self.inter_channels, 1),
            nn.MaxPool1d(2), # maxpool to reduce computation to 1/4 (see section 3.3)
        )
        self.g = nn.Sequential(
            nn.Conv1d(self.in_channels, self.inter_channels, 1),
            nn.MaxPool1d(2),
        )

        self.wz = nn.Sequential(
            nn.Conv1d(self.inter_channels, self.in_channels, 1),
            nn.BatchNorm1d(self.in_channels),
        )
        nn.init.constant_(self.wz[1].weight, 0) # zero initialization (see section 3.3)
        nn.init.constant_(self.wz[1].bias, 0)

    def forward(self, input):
        g = self.g(input).view(input.size(0),self.inter_channels,-1)
        g = g.permute(0,2,1)

        theta = self.theta(input).view(input.size(0),self.inter_channels,-1)
        theta = theta.permute(0,2,1)

        phi = self.phi(input).view(input.size(0),self.inter_channels,-1)

        f = torch.matmul(theta,phi)
        f = nn.functional.softmax(f,dim=-1)

        y = torch.matmul(f,g)
        y = y.permute(0,2,1).contiguous()
        y = y.view(input.size(0),self.inter_channels,*input.size()[2:])

        wz = self.wz(y)

        return wz+input

class U_Net(nn.Module):
    def __init__(self,inchannel,outchannel,is_attention=False,alpha=1,beta=5) -> None:
        super(U_Net,self).__init__()
        self.is_attention = is_attention
        self.beta = beta
        self.pool = nn.MaxPool1d(2,2)
        self.conv1 = conv_block(inchannel,int(64*alpha))
        if self.beta > 1:
            self.conv2 = conv_block(int(64*alpha),int(128*alpha))
            if self.beta > 2:
                self.conv3 = conv_block(int(128*alpha),int(256*alpha))
                if self.beta > 3:
                    self.conv4 = conv_block(int(256*alpha),int(512*alpha))
                    if self.beta > 4:
                        self.conv5 = conv_block(int(512*alpha),int(1024*alpha))

                        self.upconv5 = up_conv(int(1024*alpha),int(512*alpha))
                        self.decode5 = conv_block(int(1024*alpha),int(512*alpha))

                    self.upconv4 = up_conv(int(512*alpha),int(256*alpha))
                    self.decode4 = conv_block(int(512*alpha),int(256*alpha))

                self.upconv3 = up_conv(int(256*alpha),int(128*alpha))
                self.decode3 = conv_block(int(256*alpha),int(128*alpha))

            self.upconv2 = up_conv(int(128*alpha),int(64*alpha))
            self.decode2 = conv_block(int(128*alpha),int(64*alpha))

        self.decode1 = nn.Conv1d(int(64*alpha),outchannel,1,1)
        if self.is_attention:
            self.nonlocal1 = NonLocal1D(outchannel)

    def forward(self,input):
        # encoder path
        e1 = self.conv1(input) 
        if self.beta > 1:
            e2 = self.pool(e1)
            e2 = self.conv2(e2) 
            if self.beta > 2:
                e3 = self.pool(e2)
                e3 = self.conv3(e3) 
                if self.beta > 3:
                    e4 = self.pool(e3)
                    e4 = self.conv4(e4) 
                    if self.beta > 4:
                        e5 = self.pool(e4)
                        e5 = self.conv5(e5) 
                        
                        d5 = self.upconv5(e5)
                        d5 = self.decode5(torch.cat((e4,d5),dim=1)) # channel first

                        d4 = self.upconv4(d5)
                        d4 = self.decode4(torch.cat((e3,d4),dim=1))
                    else:
                        d4 = self.upconv4(e4)
                        d4 = self.decode4(torch.cat((e3,d4),dim=1))

                    d3 = self.upconv3(d4)
                    d3 = self.decode3(torch.cat((e2,d3),dim=1))
                else:
                    d3 = self.upconv3(e3)
                    d3 = self.decode3(torch.cat((e2,d3),dim=1))

                d2 = self.upconv2(d3)
                d2 = self.decode2(torch.cat((e1,d2),dim=1))
            else:
                d2 = self.upconv2(e2)
                d2 = self.decode2(torch.cat((e1,d2),dim=1))
        
            d1 = self.decode1(d2)
        else:
            d1 = self.decode1(e1)

        if self.is_attention:
            d1 = self.nonlocal1(d1)

        return torch.sigmoid(d1)