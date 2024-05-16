# Online Resource Allocation for Edge Intelligence with Colocated Model Retraining and Inference
# Prerequisites
- pytorch, numpy, thop
# Usage

This is the implementation of ORRIC using the trained model in folder `model`.

The corresponding dataset can be found at [https://zenodo.org/records/2535967](https://zenodo.org/records/2535967). Please decompress `CIFAR-10-C.tar` in the root directory to obtain the `CIFAR-10-C` folder.
```
# ORRIC implementation
python train_inference_two_model.py 
```

Others:
```
# teacher (resnet_50) training
python res50_teacher_training.py
# student (mobilenet_v2) training
python mobile_student_training.py

# measure the MACs
python MACs.py
# measure the time
python measure_time.py
# measure the performance
python performance.py
```

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
