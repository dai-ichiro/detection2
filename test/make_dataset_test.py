import os
import sys
import glob
import cv2
import torch
import mmcv
from mmtrack.apis import inference_sot, init_model
from mim.commands.download import download

argv_list = sys.argv
del argv_list[0]

class_num = len(sys.argv)
print('class count = %d'%class_num)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

video_list = []
target_name = []

for i, each_class in enumerate(argv_list):
    if os.path.isdir(each_class):
        classname_without_ext = os.path.basename(each_class)
        print('name of class%d: %s'%(i, classname_without_ext))
        target_name.append(classname_without_ext)
        video_list.append(glob.glob(os.path.join(each_class, '*')))
    else:
        classname_without_ext = os.path.splitext(os.path.basename(each_class))[0]
        print('name of class%d: %s'%(i, classname_without_ext))
        target_name.append(classname_without_ext)
        video_list.append([each_class])

for i, video in enumerate(video_list):
    print('video of class%d: '%i + ','.join(video))

out_path = 'train_data'

train_images_dir = os.path.join(out_path, 'images', 'train')
train_labels_dir = os.path.join(out_path, 'labels', 'train')

os.makedirs(train_images_dir)
os.makedirs(train_labels_dir)

init_rect_list = []

for videos in video_list:
    init_rect_list_each_class = []
    for video in videos:
        cap = cv2.VideoCapture(video)
        ret, img = cap.read()
        cap.release()

        source_window = "draw_rectangle"
        cv2.namedWindow(source_window)
        rect = cv2.selectROI(source_window, img, False, False)
        # rect:(x1, y1, w, h)
        # convert (x1, y1, w, h) to (x1, y1, x2, y2)
        rect_convert = (rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
        init_rect_list_each_class.append(rect_convert)
        cv2.destroyAllWindows()
    init_rect_list.append(init_rect_list_each_class)

# load model
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
    f.write('nc: %d'%len(target_name))
    f.write('\n')
    f.write('names: ')
    f.write('[')
    output_target_name = ['\'' + x + '\'' for x in target_name]
    f.write(', '.join(output_target_name))
    f.write(']')

print('finish making dataset.')