# scsf-detector
scale fixed detection for single class

## description

The code can be run on limited hareware.

SCSF-Net is a convolutional neural network with 70 conv layers.
It can detect single class in remote sensing image.


## installation

```
git clone https://github.com/minghuicode/scsf-detector.git
cd scsf-detector
make
```

## usage 

Input should be an image and its gsd.
Gsd means the ground sample distance of input optical remote sensing image, suppose to be 12.5 cm/pixel.

The params of SCSF-Net to detect vehicle are in `vehicle.weight`.
It can detect vehicle target in remote sensing image automatical.
Output is saved in `result` folder, both visual result and txt file is created.

```
cd scsf-detecotr
./detector
```
