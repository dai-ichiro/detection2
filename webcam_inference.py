import yolov5
from argparse import ArgumentParser
import cv2

def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', type = str, required=True, help = 'pretrained weights path')
    args = parser.parse_args()

    weights = args.weights

    model = yolov5.load(weights)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = model(img_rgb)

        result_img = cv2.cvtColor(result.render()[0], cv2.COLOR_RGB2BGR)

        cv2.imshow('result', result_img)   

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
if __name__ == '__main__':
    main()