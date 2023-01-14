from ultralytics import YOLO
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 4, help='total training epochs')
    parser.add_argument('--batch', type=int, default=16, help='total batch size')
    parser.add_argument('--weights', type = str, default = 'yolov8n.pt', help = 'initial weights path')
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch
    weights = args.weights


    model = YOLO(weights)
    model.train(
        data = 'train.yaml',
        epochs = epochs,
        batch = batch_size,
    )

if __name__ == '__main__':
    main()


        