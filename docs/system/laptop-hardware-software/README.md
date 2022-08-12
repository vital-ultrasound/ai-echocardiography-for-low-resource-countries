# Laptop computer

## Alienware 17, NVIDIA GeForce RTX 3080, 16 GB 
```
$ nvidia-smi -q

==============NVSMI LOG==============

Timestamp                                 : Thu Jul 21 21:52:47 2022
Driver Version                            : 470.129.06
CUDA Version                              : 11.4

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : NVIDIA GeForce RTX 3080 Laptop GPU
    Product Brand                         : GeForce
    Display Mode                          : Enabled
    Display Active                        : Enabled
    Persistence Mode                      : Disabled

FB Memory
        Total                             : 16116 MiB
```

## Software 
```
$ hostnamectl
 Operating System: Ubuntu 20.04.3 LTS
            Kernel: Linux 5.15.0-41-generic
      Architecture: x86-64

```

## Python packages  
See [print_system.py](../../../scripts/computer-features/print_system.py) and [README](../../../scripts/computer-features)
``` 
$ python print_system.py
Python:    3.8.11 (default, Aug  3 2021, 15:09:35) 
[GCC 7.5.0]
Platform:  Linux-5.15.0-43-generic-x86_64-with-glibc2.17
NumPy:     1.20.3
opencv:    4.6.0
PyTorch:   1.9.0
cuda_is_available: True
cuda version: 11.2
cuda.device_count  1
```
