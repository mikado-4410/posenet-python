## PoseNet Python

This repository originates from [rwightman/posenet-python](https://github.com/rwightman/posenet-python) and has been 
heavily refactored to: 
 * make it run the posenet v2 networks 
 * get it to work with the latest tfjs graph serialization 
 * extend it with the ResNet50 network
 * make the code run on TF 2.x
 * get all code running in docker containers for ease of use and installation (no conda necessary)

This repository contains a pure Python implementation (multi-pose only) of the Google TensorFlow.js Posenet model. 
For a (slightly faster) PyTorch implementation that followed from this, 
see (https://github.com/rwightman/posenet-pytorch)
  

### Install

A suitable Python 3.x environment with Tensorflow 2.x.

If you want to use the webcam demo, a pip version of opencv (`pip install opencv-python`) is required instead of 
the conda version. Anaconda's default opencv does not include ffpmeg/VideoCapture support. Also, you may have to 
force install version 3.4.x as 4.x has a broken drawKeypoints binding.

Have a look at the docker configuration for a quick setup. If you want conda, have a look at the `requirements.txt` 
file to see what you should install.

### Using Docker 

A convenient way to run this project is by building and running the docker image, because it has all the requirements 
built-in. 
The GPU version is tested on a Linux machine. You need to install the nvidia host driver and the nvidia-docker toolkit. 
Once set up, you can make as many images as you want with different dependencies without touching your host OS 
(or fiddling with conda).  

If you just want to test this code, you can run everything on a CPU just as well. You still get 8fps on mobilenet and 
4fps on resnet50. Replace `GPU` below with `CPU` to test on a CPU.

```bash
./docker_img_build.sh GPU 
. ./exportGPU.sh
./get_test_images_run.sh
./image_demo_run.sh
``` 

Some pointers to get you going on the Linux machine setup. Most links are based on Ubuntu, but other distributions 
should work fine as well. 
* [Install docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/ )
* [Install the NVIDIA host driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)
  * remember to reboot here
* [Install the NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* check your installation: `docker run --gpus all nvidia/cuda nvidia-smi`


### Usage

There are three demo apps in the root that utilize the PoseNet model. They are very basic and could definitely be 
improved.

The first time these apps are run (or the library is used) model weights will be downloaded from the TensorFlow.js 
version and converted on the fly.

#### image_demo.py 

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton 
overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

#### benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is 
run `--num_images` times with no drawing and no text output.

#### webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and 
skeletons and rendered to the screen. The default args for the webcam_demo assume device_id=0 for the camera and 
that 1280x720 resolution is possible.

### Credits

The original model, weights, code, etc. was created by Google and can be found at 
https://github.com/tensorflow/tfjs-models/tree/master/posenet

This port is initially created by Ross Wightman and later upgraded by Peter Rigole and is in no way related to Google.

The Python conversion code that started me on my way was adapted from the CoreML port at 
https://github.com/infocom-tpo/PoseNet-CoreML

### TODO 
* Performance improvements (especially edge loops in 'decode.py')
* OpenGL rendering/drawing
* Comment interfaces, tensor dimensions, etc
