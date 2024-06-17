# IP2P_Style_Transfer
## Installation
### Git cloning
```
git clone https://github.com/Yenhongxuan/IP2P_Style_Transfer.git

cd IP2P_Style_Transfer

git submodule update --init --recursive
```

### Download pre-trained weight for instruct-pix2pix

```
cd instruct-pix2pix

bash scripts/download_checkpoints.sh

cd ..
```


### Download pre-trained weight for background style transfer
The trained-weights and data our system needs are stored [here](https://drive.google.com/drive/folders/1ZGiSLfpzYJQ050VhV8kYr7nuAatot-Rj?usp=sharing). 
```
cd RealTime_Video_Style_Transfer
```

The weights **rvm_mobilenetv3.pth** should be put under **/mat/checkpoints/**, **style_net-TIP-final.pth** should be put under **/style_transfer/test/Model/**.
```
cd ..
```


### Set up YOLOv8 instance segmentation
```
cd RealTime_Video_Style_Transfer
git clone https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation.git
mv ONNX-YOLOv8-Instance-Segmentation/ ONNX_YOLOv8_Instance_Segmentation/
cd ..
```
Also, please the **from yoloseg.utils import xywh2xyxy, nms, draw_detections, sigmoid** in **ONNX_YOLOv8_Instance_Segmentation/yoloseg/YOLOSeg.py** with **from ONNX_YOLOv8_Instance_Segmentation.yoloseg.utils import xywh2xyxy, nms, draw_detections, sigmoid**

### Download pre-trained weight for YOLOv8
```
cd RealTime_Video_Style_Transfer
```
The pre-trained weight are store [here](https://drive.google.com/drive/folders/1njB96mxyCT9GjZxMHg7kV065zhJ0rd6h?usp=sharing). Please download it and put it under **ONNX_YOLOv8_Instance_Segmentation/models**


### Download the source video
```
cd videos
```
The source videos for test are stored [here](https://drive.google.com/drive/folders/1aiI7Gcje7Il3oik1_EAgfeK5gFFuOj3J?usp=sharing).
Please download the videos into the folder "videos". 
```
cd ..
``` 

## Build environment
There are two environment used in the process
### RTVST
```
conda env create -f envs/RTVST.yml
```

### ip2p
```
conda env create -f envs/ip2p.yml
```


## Running experiment
### Pre-process and segment foreground and background


