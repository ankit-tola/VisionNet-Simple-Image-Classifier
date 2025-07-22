import torch
from model import CNN
from data_loader import get_data
from train import train
from test import test
from utils import get_writer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CNN()
    train_loader, test_loader = get_data(batch_size=64)
    writer = get_writer()

    train(model, train_loader, device, writer)
    test(model, test_loader, device)

if __name__ == "__main__":
    main()
