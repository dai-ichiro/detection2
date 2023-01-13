import yolov5
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='test image path')
    parser.add_argument('--weights', type = str, required=True, help = 'pretrained weights path')
    args = parser.parse_args()

    img_path = args.image
    weights = args.weights

    model = yolov5.load(weights)
    result = model(img_path)
    result.show()
    
if __name__ == '__main__':
    main()


        
