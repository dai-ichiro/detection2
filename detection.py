import os
import glob
import cv2
import torch
import mmcv
from ultralytics import YOLO
from mmtrack.apis import inference_sot, init_model
from mim.commands.download import download

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--videos_dir', type=str, default='videos', help='video folder name' )
parser.add_argument('--epochs', type=int, default=4, help='total training epochs')
parser.add_argument('--batch', type=int, default=8, help='total batch size')
parser.add_argument('--weights', type = str, default = 'yolov5s.pt', help = 'initial weights path')
args = parser.parse_args()

videos_dir = args.videos_dir
epochs = args.epochs
batch_size = args.batch
weights = args.weights

def tracking():
    class_list = glob.glob(os.path.join(videos_dir, '*'))

    class_num = len(class_list)
    print(f'class count = {class_num}')

    video_list = []
    classname_list = []

    for i, each_class in enumerate(class_list):
        if os.path.isdir(each_class):
            classname_without_ext = os.path.basename(each_class)
            classname_list.append(classname_without_ext)
            video_list.append(glob.glob(os.path.join(each_class, '*')))
        else:
            classname_without_ext = os.path.splitext(os.path.basename(each_class))[0]
            classname_list.append(classname_without_ext)
            video_list.append([each_class])

    for i, classname in enumerate(classname_list):
        print(f'class {i}: {classname}')

    for i, videos_in_each_class in enumerate(video_list):
        videos_str = ', '.join(videos_in_each_class)
        print(f'videos of class {i}: {videos_str}')
    
    out_path = 'train_data'

    train_images_dir = os.path.join(out_path, 'images', 'train')
    train_labels_dir = os.path.join(out_path, 'labels', 'train')

    os.makedirs(train_images_dir)
    os.makedirs(train_labels_dir)
    
    init_rect_list = []

    for videos_in_each_class in video_list:
        temporary_list_each_class = []
        for video in videos_in_each_class:
            cap = cv2.VideoCapture(video)
            _, img = cap.read()
            cap.release()

            source_window = "draw_rectangle"
            cv2.namedWindow(source_window)
            rect = cv2.selectROI(source_window, img, False, False)
            # rect:(x1, y1, w, h)
            # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            rect_convert = (rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
            temporary_list_each_class.append(rect_convert)
            cv2.destroyAllWindows()
        init_rect_list.append(temporary_list_each_class)

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('models', exist_ok=True)
    checkpoint_name = 'siamese_rpn_r50_20e_lasot'
    checkpoint = download(package='mmtrack', configs=[checkpoint_name], dest_root="models")[0]
    model = init_model(os.path.join('models', checkpoint_name + '.py'), os.path.join('models', checkpoint), device=device)

    print('start making dataset...')
    # tracking
    for class_index, videos in enumerate(video_list):
        for video_index, video in enumerate(videos):
            # read video
            frames = mmcv.VideoReader(video)
            h = frames.height
            w = frames.width
            # tracking
            for frame_index, frame in enumerate(frames):
                result = inference_sot(model, frame, init_rect_list[class_index][video_index], frame_id=frame_index)
                bbox = result['track_bboxes']
                # bbox:(x1, y1, x2, y2)
                center_x = ((bbox[0] + bbox[2]) / 2) / w
                center_y = ((bbox[1] + bbox[3]) / 2) / h
                width = (bbox[2] - bbox[0]) / w
                height = (bbox[3] - bbox[1]) /h

                filename = '%d_%d_%06d'%(class_index, video_index, frame_index)

                # save image
                jpeg_filename = filename + '.jpg'
                cv2.imwrite(os.path.join(train_images_dir, jpeg_filename), frame)

                # save text
                txt_filename = filename + '.txt'
                with open(os.path.join(train_labels_dir, txt_filename), 'w') as f:
                    f.write('%d %f %f %f %f'%(class_index, center_x, center_y, width, height))

    abs_out_path = os.path.abspath(out_path)

    with open('train.yaml', 'w', encoding='cp932') as f:
        #f.write('path: %s'%out_path)
        #f.write('\n')
        f.write(f'train: {abs_out_path}/images/train')
        f.write('\n')
        f.write(f'val: {abs_out_path}/images/train')
        f.write('\n')
        f.write('nc: %d'%class_num)
        f.write('\n')
        f.write('names: ')
        f.write('[')
        output_target_name = ['\'' + x + '\'' for x in classname_list]
        f.write(', '.join(output_target_name))
        f.write(']')
'''
def yolo_train():
    train.run(data='train.yaml', 
        epochs = epochs,
        batch_size = batch_size,
        weights = weights)
'''
def yolo_train():
    model = YOLO('yolov8n.pt')
    model.train(
        data = 'train.yaml',
        epochs = epochs
    )
if __name__ == '__main__':
    tracking()
    print('finish making dataset.')
    print('start training...')
    yolo_train()
    print('finish training.')

