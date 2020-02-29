import torch
from efficient_unet.efficientnet import EfficientNet
from efficient_unet.efficientunet import get_efficientunet_b0, get_efficientunet_b1, get_efficientunet_b2, get_efficientunet_b3

# This is a demo to show you how to use the library.
t = torch.rand(2, 3, 224, 224).cuda()

# EfficientNet test
# model = EfficientNet.from_name('efficientnet-b5', n_classes=5, pretrained=False).cuda()
# print(model(t).size())
#
# # EfficientNet with custom head test
# model_ch = EfficientNet.custom_head('efficientnet-b5', n_classes=5, pretrained=False).cuda()
# print(model_ch(t).size())

# EfficientUnet test
b0unet = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=False).cuda()
print(b0unet(t).size())
