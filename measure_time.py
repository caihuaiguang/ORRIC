# https://deci.ai/blog/measure-inference-time-deep-neural-networks/

# Import necessary libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from torch.utils.data import random_split
from thop import profile
 
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2

# for batch_size in [2,10,20,40,100,200,300,400,500,600,700,800,900,1000]:
for batch_size in [1000]:
    print(batch_size)  
    Inference_configuration_list = np.asarray([20,24,28,32])
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # model = mobilenet_v2().to(device)
    # model.load_state_dict(torch.load("models/mobile_student_model.pth"))  
    model = resnet50().to(device)
    model.load_state_dict(torch.load("models/res50_model.pth")) 
    for i in Inference_configuration_list:  
        dummy_input = torch.randn(batch_size, 3,int(i),int(i), dtype=torch.float).to(device)

        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(10):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time/batch_size

        mean_syn = round(1000*np.sum(timings) / repetitions,2)
        std_syn = np.std(timings)
        print(i,"*",i,": ",mean_syn, "us")


