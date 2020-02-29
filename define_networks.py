import torch.nn as nn
from efficient_unet.efficientunet import get_efficientunet_b0, get_efficientunet_b1, get_efficientunet_b2, get_efficientunet_b3

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        # t = torch.rand(2, 3, args.img_size, args.img_size).cuda()
        if args.backbone == 'efficientunet_b0':
            self.model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False) # out_channels=2
        elif args.backbone == 'efficientunet_b1':
            self.model = get_efficientunet_b1(out_channels=1, concat_input=True, pretrained=False)
        elif args.backbone == 'efficientunet_b2':
            self.model = get_efficientunet_b2(out_channels=1, concat_input=True, pretrained=False)
        elif args.backbone == 'efficientunet_b3':
            self.model = get_efficientunet_b3(out_channels=1, concat_input=True, pretrained=False)
        else:
            print('Invalid backbone args input. Please check again.')
            raise NotImplementedError
        self.model.to('cuda')
        print('Generator transfered to CUDA device.')
        # print(self.model(t).size())

    def forward(self, z):
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.batch_size = self.args.batch_size
        self.in_channels = args.channels

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(self.in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.model.to('cuda')
        print('Discriminator transfered to CUDA device.')

    def forward(self, img):
        # img_flat = img.view(img.shape[0], -1)
        # print('shape of img in define_networks.py:', img.shape)
        validity = self.model(img)
        # print('shape of validity:', validity.shape)
        return validity