### Device initialization  
Rememeber that openacc has an overhead needed to initialize the device.  
In your codes, before writing the various pragmas, manually initialize the gpu by writing *only once*:  
```acc_init(acc_device_nvidia)```
