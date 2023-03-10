# detection (Siamese + YOLOv8)

For more details

https://touch-sp.hatenablog.com/entry/2023/01/14/122603

## Environment
~~~
Windows 11
CUDA 11.6.2
Python 3.9.13
~~~

## Requirements
~~~
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html
pip install -r https://raw.githubusercontent.com/dai-ichiro/detection2/main/requirements.txt
~~~

## How to use
### Clone
~~~
git clone https://github.com/dai-ichiro/detection2.git
cd detection2
~~~
### Training
~~~
python tools/download_sample_videos.py
python detection.py
~~~
### Inference after training
#### image file
~~~
python inference.py --image sample_images/sample01.jpg --weights runs/detect/train/weights/best.pt
~~~
#### webcam
~~~
python webcam_inference.py --weights runs/detect/train/weights/best.pt
~~~
