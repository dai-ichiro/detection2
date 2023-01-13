# detection (Siamese + YOLOv5)

For more details

https://touch-sp.hatenablog.com/entry/2022/09/19/120221

## Environment
~~~
Windows 11
CUDA 11.6.2
Python 3.9.13
~~~

## Requirements
~~~
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
pip install -r https://raw.githubusercontent.com/dai-ichiro/detection/main/requirements.txt
~~~

## How to use
### Clone
~~~
git clone https://github.com/dai-ichiro/detection.git
cd detection
~~~
### Training
~~~
python tools/download_sample_videos.py
python detection.py
~~~
### Inference after training
~~~
python inference.py --image sample_images/sample01.jpg --weights runs/train/exp/weights/best.pt
~~~
