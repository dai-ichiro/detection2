from ultralytics import YOLO
import os
from PIL import Image
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='test image path')
    parser.add_argument('--weights', type = str, required=True, help = 'pretrained weights path')
    args = parser.parse_args()

    img_path = args.image
    weights = args.weights

    model = YOLO(weights)  
    model(img_path, save=True)

    img_fname = os.path.basename(img_path)
    pil = Image.open(os.path.join('runs', 'detect', 'predict', img_fname))
    pil.show()
    
if __name__ == '__main__':
    main()


        
