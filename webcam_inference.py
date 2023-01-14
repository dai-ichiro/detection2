from ultralytics import YOLO
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', type = str, required=True, help = 'pretrained weights path')
    args = parser.parse_args()

    weights = args.weights
    
    model = YOLO(weights)
    
    model(0, show=True)
    
if __name__ == '__main__':
    main()
