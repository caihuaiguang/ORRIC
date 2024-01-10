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

    """
    mobilenet
(chg) [caihuaiguang@earth /home/caihuaiguang/inference_training/EATA/CIFAR]$ python measure_time.py
8
8 * 8 :  5.106866456667582 ms
16
16 * 16 :  5.151488223075867 ms
24
24 * 24 :  5.261584328015645 ms
32
32 * 32 :  5.256425499916077 ms
    """

    """
(chg) [caihuaiguang@earth /home/caihuaiguang/inference_training/EATA/CIFAR]$ python measure_time.py
2
8 * 8 :  3.2773171242078147 ms
16 * 16 :  3.311416478951772 ms
24 * 24 :  3.6138021834691365 ms
32 * 32 :  3.624360053539276 ms
10
8 * 8 :  0.6729593504269918 ms
16 * 16 :  0.7138318279584249 ms
24 * 24 :  0.7441065815289815 ms
32 * 32 :  0.7370296120643616 ms
20
8 * 8 :  0.34066226609547934 ms
16 * 16 :  0.37545741399129234 ms
24 * 24 :  0.3825810718536377 ms
32 * 32 :  0.3659529814720154 ms
40
8 * 8 :  0.1748962641954422 ms
16 * 16 :  0.1885815518697103 ms
24 * 24 :  0.21061846474806467 ms
32 * 32 :  0.2973395066261292 ms
100
8 * 8 :  0.07330056756337484 ms
16 * 16 :  0.092412615776062 ms
24 * 24 :  0.1803245028177897 ms
32 * 32 :  0.28461042849222823 ms
200
8 * 8 :  0.035724879423777264 ms
16 * 16 :  0.08002931909561156 ms
24 * 24 :  0.16773035411834716 ms
32 * 32 :  0.302846466700236 ms
300
8 * 8 :  0.02904405724737379 ms
16 * 16 :  0.07570817582872179 ms
24 * 24 :  0.168762697813246 ms
32 * 32 :  0.29794098705715605 ms
400
8 * 8 :  0.024449937375386553 ms
16 * 16 :  0.07586070982615152 ms
24 * 24 :  0.16687061398824057 ms
32 * 32 :  0.29272735595703125 ms
500
8 * 8 :  0.024075131085713703 ms
16 * 16 :  0.07519986373901368 ms
24 * 24 :  0.16538673950195312 ms
32 * 32 :  0.2677613763427734 ms
600
8 * 8 :  0.023244684049818252 ms
16 * 16 :  0.07181462828318279 ms
24 * 24 :  0.16518468657599555 ms
32 * 32 :  0.2690322490268284 ms
700
8 * 8 :  0.023476622626894993 ms
16 * 16 :  0.07178724305289132 ms
24 * 24 :  0.1622289850507464 ms
32 * 32 :  0.2648680366879418 ms
800
8 * 8 :  0.022513290643692018 ms
16 * 16 :  0.07310153926213582 ms
24 * 24 :  0.16210014298756917 ms
32 * 32 :  0.2652834977467855 ms
900
8 * 8 :  0.02319671180866383 ms
16 * 16 :  0.07231843708179615 ms
24 * 24 :  0.16365961755823208 ms
32 * 32 :  0.2655107862684462 ms
1000
8 * 8 :  0.022916625200907387 ms
16 * 16 :  0.0712015345509847 ms
24 * 24 :  0.16197738118489582 ms
32 * 32 :  0.26438268951416016 ms

    """

