from yolov5 import train 
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 4, help='total training epochs')
    parser.add_argument('--batch', type=int, default=16, help='total batch size')
    parser.add_argument('--weights', type = str, default = 'yolov5s.pt', help = 'initial weights path')
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch
    weights = args.weights

    train.run(data='train.yaml', 
    batch_size = batch_size,
    epochs = epochs,
    weights = weights)

if __name__ == '__main__':
    main()


        