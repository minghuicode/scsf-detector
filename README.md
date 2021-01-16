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
Gsd means the ground sample distance of an optical remote sensing(ORS) image.
In SCSF-Net, the default setting of gsd is 12.5 cm/pixel.

1. Start Detection

```
cd scsf-detecotr
./detector
```
Model will load paramters in file `vehicle.weight`. 

2. Iteration

In each iteartions, input the path of an image. And input its gsd in below row.

```
gsd-12.5/testA.png
12.5
```

It may take few seconds to run the detection result of vehicle target.
Output is saved in `result` folder, both visual result and txt file is created.

3. Exit

In each iterations, if input file is not found, this process will exit.
