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
Also, please replace the **from yoloseg.utils import xywh2xyxy, nms, draw_detections, sigmoid** in **ONNX_YOLOv8_Instance_Segmentation/yoloseg/YOLOSeg.py** with **from ONNX_YOLOv8_Instance_Segmentation.yoloseg.utils import xywh2xyxy, nms, draw_detections, sigmoid**

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
In our experiment, we use Mambaforge to build the environment. Please use the [link](https://github.com/conda-forge/miniforge/releases) to install Mambaforge first. 


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
First, we extract foreground and background using YOLOv8 and apply style transfer on the background conditioned on styled image. 
```
cd RealTime_Video_Style_Transfer

conda activate RTVST

python data_preprocess.py --input ../videos/video_0.mp4

cd ..
```
You can test on other video by assign path of video to property --input.
Optional, there are several styled image can be the image condition in **./style_transfer/inputs/styles/**. Below are an example of tesing on different video and using different conditioned image. 
```
python data_preprocess.py --input "path of video" --style_bg ./style_transfer/inputs/styles/starry_night.jpg
```

After running above commands, the styled background and foreground of each frames will be saved to IP2P_Style_Transfer/results with folder name video_{index of video}. 

### Apply instruct-pix2pix on foreground
First,  we need to generate the bash script to run. Second, run the bash script. 
```
cd instruct-pix2pix

conda activate ip2p

python generate_script.py --video ../results/video_0 --prompt "Turn them into clowns"

bash ./running_scripts/video_0.sh

cd ..
```
The generated bash script file will be saved to **./running_scripts**. Please the corresponding bash script to apply instruct-pix2pix. After running the bash script, the styled foreground image will be saved to the specified video folder. 

You can choose the video folder by assign path of video folder to the property **--video**. Also, you can using different prompt by assign different command to the property **--prompt**. For example, 
```
cd instruct-pix2pix

conda activate ip2p

python generate_script.py --video ../results/video_{idx} --prompt "Turn them into clowns"

bash ./running_scripts/video_{idx}.sh

cd ..
```

### Merge the styled foreground and background
Finally, we need to merge the styled foreground and background into a video. 
```
cd RealTime_Video_Style_Transfer

conda activate RTVST

python merge.py --video ../results/video_0

cd ..
```
You can specify the video folder you want to merge by assign path of folder to the property **--video**. The result styled video will be saved at the given video folder. 
