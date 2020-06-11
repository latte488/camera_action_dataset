import datasets
import models
import torch
import argparse
import csv 
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import shutil

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()

    train_loader, test_loader = datasets.prepare(batch_size=args.batch_size)
    model = models.Net(num_classes=8).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    log_dir = 'data/runs'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)
    else:
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    epoch_digit = len(list(str(args.epoch)))
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        train_acc = 0
        train_number = 0
        for batch_i, (inputs, labels) in tqdm(enumerate(train_loader), desc='Train', total=len(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_number += labels.size(0)
            train_acc += predicted.eq(labels).sum().item()

        train_loss /= batch_i + 1
        train_acc = 100. * train_acc / train_number
        writer.add_scalar('Train-Loss', train_loss, epoch + 1)
        writer.add_scalar('Train-Accuracy', train_acc, epoch + 1)
        print(f'Train : Epoch {epoch + 1:{epoch_digit}} | Loss {train_loss:.3f} | Acc {train_acc:.3f}')

        model.eval()
        test_loss = 0
        test_acc = 0
        test_number = 0
        with torch.no_grad():
            for batch_i, (inputs, labels) in tqdm(enumerate(test_loader), desc='Test', total=len(test_loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_number += labels.size(0)
                test_acc += predicted.eq(labels).sum().item()

        test_loss /= batch_i + 1
        test_acc = 100. * test_acc / test_number
        writer.add_scalar('Test-Loss', test_loss, epoch + 1)
        writer.add_scalar('Test-Accuracy', test_acc, epoch + 1)
        print(f'Test : Epoch {epoch + 1:{epoch_digit}} | Loss {test_loss:.3f} | Acc {test_acc:.3f}')

    writer.close()
