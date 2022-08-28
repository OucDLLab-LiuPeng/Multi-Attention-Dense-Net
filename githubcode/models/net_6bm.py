# Trained From NewEMEnhanceWithTrans_1114_G_340
# Trained From NewEMEnhanceWithTrans_1126_G_80
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import init
import functools
from torch.optim import lr_scheduler

# Actually we don't use Single Dense Layer in this model.
class SingleDenseLayer(nn.Module):
	def __init__(self, input_dim=3, growthRate=12):
		super(SingleDenseLayer, self).__init__()
		self.activation = nn.ELU(inplace=True)
		self.bn = nn.BatchNorm2d(num_features=input_dim)
		self.conv1 = nn.Conv2d(in_channels=input_dim,out_channels=growthRate,kernel_size=3,stride=1,padding=1,bias=False)
	def forward(self, x):
		output = self.conv1(self.activation(self.bn(x)))
		output = torch.cat((x,output),1)
		return output

# Each Dense Block in our model is made of 5 Bottle Neck Layers.
class BottleNeckLayer(nn.Module):
	def __init__(self, input_dim=3, growthRate=12):
		super(BottleNeckLayer, self).__init__()
		self.activation = nn.ReLU(inplace=True)
		# self.activation = nn.ELU(inplace=True)
		self.bn = nn.BatchNorm2d(num_features=int(input_dim))
		self.conv1 = nn.Conv2d(in_channels=input_dim,out_channels=4*growthRate,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv2 = nn.Conv2d(in_channels=4*growthRate,out_channels=growthRate,kernel_size=3,stride=1,padding=1,bias=False)
	def forward(self, x):
		output = self.conv1(self.activation(self.bn(x)))
		output = self.conv2(output)
		output = torch.cat((x,output),1)
		return output

# The Transition Layer connects 2 Dense block.
class TransitionLayer(nn.Module):
	def __init__(self, input_dim, output_dim, is_pool=False):
		super(TransitionLayer, self).__init__()
		self.activation = nn.ReLU(inplace=True)
		self.bn = nn.BatchNorm2d(num_features=input_dim)
		self.conv = nn.Conv2d(in_channels=input_dim,out_channels=output_dim,kernel_size=1,stride=1,padding=0,bias=False)
		self.pool = nn.AvgPool2d(kernel_size=2,stride=2)
		self.is_pool = is_pool
	def forward(self, x):
		result = self.conv(self.activation(self.bn(x)))
		if(self.is_pool):
			result = self.pool(result)
		return result

class DenseBlock(nn.Module):
	def __init__(self, input_dim=3, growthRate=12, num_dense_layers=5):
		super(DenseBlock, self).__init__()
		
		self.im = input_dim
		layers = []
		for i in range(num_dense_layers):
			layers.append(BottleNeckLayer(input_dim+i*growthRate,growthRate))
		self.dense = nn.Sequential(*layers)
	
	def forward(self, x):
		return self.dense(x)

# ****** Define Unet ******
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
		
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ****** Define FCN-8 ******

class FCN8(nn.Module):

    def __init__(self, in_channel=3, numclass=3):
        super(FCN8, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=3)
        self.dropout1 = nn.Dropout(0.85)

        self.conv7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.85)

        self.conv8 = nn.Conv2d(in_channels=4096, out_channels=numclass, kernel_size=1)

        self.tranconv1 = nn.ConvTranspose2d(in_channels=numclass, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.tranconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=2, output_padding=1)

        self.tranconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=numclass, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.conv12(self.conv11(x))
        x = self.pool1(x)
        x = self.pool2(self.conv22(self.conv21(x)))
        x1 = self.pool3(self.conv33(self.conv32(self.conv31(x))))
        x2 = self.pool4(self.conv43(self.conv42(self.conv41(x1))))
        x = self.pool5(self.conv53(self.conv52(self.conv51(x2))))
        x = self.dropout1(self.conv6(x))
        x = self.dropout2(self.conv7(x))
        x = self.conv8(x)
        x = self.tranconv1(x)
        x = x2 + x
        x = self.tranconv2(x)
        x = x1 + x
        x = self.tranconv3(x)
        return x


class Downsample(nn.Module):
	def __init__(self, in_channels, out_channels, scale):
		super(Downsample, self).__init__()
		layers = []
		num_channels = in_channels
		for i in range(int(math.log(scale, 2))-1):
			layers.append(Down(num_channels, num_channels*2))
			num_channels = num_channels * 2
		layers.append(Down(num_channels, out_channels))
		self.downsample = nn.Sequential(*layers)
		

	def forward(self, x):
		return self.downsample(x)
class Upsample(nn.Module):
	def __init__(self, in_channels, out_channels, scale):
		super(Upsample, self).__init__()
		layers = []
		num_channels = in_channels
		for i in range(int(math.log(scale, 2))-1):
			layers.append(nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels//2, kernel_size=4, stride=2, padding=1))
			num_channels = num_channels // 2
		layers.append(nn.ConvTranspose2d(in_channels=num_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
		self.upsample = nn.Sequential(*layers)
		

	def forward(self, x):
		return self.upsample(x)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv00 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, bias=True)
        self.conv01 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, bias=True)

        self.down1 = nn.MaxPool2d(2)
        self.conv11 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, bias=True)
        self.up11 = Upsample(channel, channel, 2)
        self.down2 = nn.MaxPool2d(2)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, bias=True)
        self.up21 = Upsample(channel, channel, 4)
        self.conv_out = nn.Conv2d(in_channels=3*channel, out_channels=channel, kernel_size=3, padding=1, bias=True)
    def forward(self, x):
        x00 = self.activation(self.conv00(x)) # 16
        x01 = self.activation(self.conv01(x00)) # 16
        x0_out = self.sigmoid(x01)
        x0_out = x * x0_out
        
        x11 = self.down1(x00) # 16
        x11 = self.activation(self.conv11(x11)) # 16
        x1_out = self.up11(x11) # 16
        x1_out = self.sigmoid(x1_out)
        x1_out = x * x1_out
        x21 = self.down2(x11) # 16
        x21 = self.activation(self.conv21(x21)) # 16
        x2_out = self.up21(x21) # 16
        x2_out = self.sigmoid(x2_out)
        x2_out = x * x2_out
        y = torch.cat([x0_out, x1_out, x2_out], dim=1)
        y = self.activation(self.conv_out(y)) # 16
        return y

class UniversalEstimator(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear, sigmoid):
        super(UniversalEstimator, self).__init__()
        if sigmoid:
            self.estimate = nn.Sequential(
                    UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear),
                    PALayer(n_classes),
                    nn.Sigmoid()
            )
        else:
            self.estimate = nn.Sequential(
                    Unet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear),
                    PALayer(n_classes)
            )

    def forward(self, x):
        y = self.estimate(x)
        return y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.ca = nn.Sequential(
                nn.AvgPool2d(16),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ca(x)
        y = self.avg_pool(y)
        y = self.sigmoid(y)
        return x * y

# ****** Define Generator and Discriminator ******
class AtmLightEstimator(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(AtmLightEstimator, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False)
		self.bn = nn.BatchNorm2d(num_features=in_channels)
		self.activation=nn.LeakyReLU(0.2, inplace=True)
		self.down1 = Downsample(out_channels, out_channels*2, 2)
		self.down2 = Downsample(out_channels*2, out_channels*4, 2)
		self.down3 = Downsample(out_channels*4, out_channels*8, 2)
		self.up1 = Upsample(out_channels*8, out_channels*4, 2)
		self.up2 = Upsample(out_channels*8, out_channels*4, 2)
		self.up3 = Upsample(out_channels*6, out_channels*3, 2)
		self.ca = CALayer(out_channels*3)
		self.pa = PALayer(out_channels*3)
		self.transition = TransitionLayer(out_channels*3, out_channels, False)
		self.bypass = TransitionLayer(out_channels, out_channels, False)

	def forward(self, x):
		x = self.conv(self.activation(self.bn(x)))
		feature2x = self.down1(x) # in_channels*2
		feature4x = self.down2(feature2x) #in_channels*4
		feature8x = self.down3(feature4x) #in_channels*8
		output = self.up1(feature8x) #in_channels*4
		output = self.up2(torch.cat((output, feature4x),1)) #in_channels*4
		output = self.up3(torch.cat((output, feature2x),1))
		A = self.ca(output)
		A = self.pa(A)
		A = self.transition(A) + self.bypass(x)
		# shape_out1 = A.data.size()
		# shape_out = shape_out1[2:4]
		# 
		# A = F.avg_pool2d(A, shape_out1[2])
		# A = self.upsample(self.activation(A),size=shape_out)
		return A
		
class dehaze_net(nn.Module):

	def __init__(self, input_dim=3, growthRate=12, num_dense_layers=5,reduction=0.5):
		super(dehaze_net, self).__init__()

		self.t_estimator = nn.Sequential(
                    UNet(n_channels=input_dim, n_classes=3, bilinear=False),
                    nn.Sigmoid()
            )

		self.A_estimator = nn.Sequential(
                    UNet(n_channels=input_dim, n_classes=3, bilinear=False),
                    nn.Sigmoid()
            )
		
	def forward(self, x, enable_A=1, true_A=0, enable_t=1, true_t=1):
		# if flag = 1, use true_A and abandon trained A.
		t = self.t_estimator(x)
		A = self.A_estimator(x)

		clean_image = (x-A)/t + A
		
		return clean_image, t, A

class enhance_net(nn.Module):

    def __init__(self, input_dim=3, growthRate=12, num_dense_layers=5):
        super(enhance_net, self).__init__()

        num_channels = input_dim
        self.conv1 = nn.Conv2d(num_channels, 20, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(20, 40, 1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(40, 60, 1, padding=0, bias=True)
        self.EM1 = nn.Sequential(
                    DenseBlock(60, growthRate, num_dense_layers),
                    TransitionLayer(60+growthRate*num_dense_layers, 60),
                    PALayer(60),
                    CALayer(60)
            )
        self.EM2 = nn.Sequential(
                    DenseBlock(60, growthRate, num_dense_layers),
                    TransitionLayer(60+growthRate*num_dense_layers, 60),
                    PALayer(60),
                    CALayer(60)
            )
        self.EM3 = nn.Sequential(
                    DenseBlock(60, growthRate, num_dense_layers),
                    TransitionLayer(60+growthRate*num_dense_layers, 60),
                    PALayer(60),
                    CALayer(60)
            )
        self.EM4 = nn.Sequential(
                    DenseBlock(60, growthRate, num_dense_layers),
                    TransitionLayer(60+growthRate*num_dense_layers, 60),
                    PALayer(60),
                    CALayer(60)
            )
        self.EM5 = nn.Sequential(
                    DenseBlock(60, growthRate, num_dense_layers),
                    TransitionLayer(60+growthRate*num_dense_layers, 60),
                    PALayer(60),
                    CALayer(60)
            )
        self.EM6 = nn.Sequential(
                    DenseBlock(60, growthRate, num_dense_layers),
                    TransitionLayer(60+growthRate*num_dense_layers, 60),
                    PALayer(60),
                    CALayer(60)
            )
        self.conv4 = nn.Conv2d(60, 40, 1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(40, 20, 1, padding=0, bias=True)
        self.conv6 = nn.Conv2d(20, num_channels, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.relu(self.conv1(x))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        
        output = self.EM1(output)
        output = self.EM2(output)
        output = self.EM3(output)
        output = self.EM4(output)
        output = self.EM5(output)
        output = self.EM6(output)
        
        output = self.relu(self.conv4(output))
        output = self.relu(self.conv5(output))
        output = self.relu(self.conv6(output))
        print("--------------------------------")
        return output
        
ndf = 64
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)






