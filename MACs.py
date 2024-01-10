import torch
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2
from thop import profile
from thop import clever_format 
# model = mobilenet_v2()
model = resnet50()
for i in [20,24,28,32]:
    input = torch.randn(1, 3, i, i)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.2f")
    print("i, macs, params", i, macs, params)