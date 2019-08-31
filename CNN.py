import math
import numpy as np
from PIL import Image
import os
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchsummary import summary



def load_images(image_size=100, batch_size=4, root=r"C:\Users\Paul\Desktop\Dataset\Dataset"):

    transform = transforms.Compose([
                    transforms.Resize((100,100)),
                    transforms.ToTensor()])

    train_set = dsets.ImageFolder(root=root,transform=transform)
    train_size = int(0.7 * len(train_set))
    dev_size = int(0.2 * len(train_set))
    test_size = len(train_set) - train_size - dev_size
    train_dataset, test_dataset, dev_dataset = torch.utils.data.random_split(train_set, [train_size, test_size, dev_size])

    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader, test_loader, dev_loader

class CNN_ex(nn.Module):
    def __init__(self):
        super(CNN_ex,self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 8, padding=0, stride=1, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(stride=1, padding=0, kernel_size=4))
        self.l2 = nn.Sequential(
            nn.Conv2d(8, 5, padding=0, stride=1, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool2d(stride=1, padding=0, kernel_size=5))
        self.l3 = nn.Sequential(
            nn.Conv2d(5, 5, padding=0, stride=1, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool2d(stride=5, padding=0, kernel_size=4))
        self.l4 = nn.Linear(16*16*5,100)
        self.l5 = nn.Linear(100,60)
        self.l6 = nn.Linear(60, 10)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out.reshape(out.size(0), -1))
        out = self.l5(out)
        out = self.l6(out)
        return out


if __name__ == '__main__':
    batch_size = 16
    epochs = 30
    learning_rate = 0.01

    X_train, X_test, X_dev = load_images(image_size=100,batch_size=batch_size)
    cnn = CNN_ex()
    cnn_param = cnn.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_param)

    trainer = create_supervised_trainer(cnn, optimizer, criterion)
    evaluator = create_supervised_evaluator(cnn, metrics = {'accuracy': Accuracy(), 'nll': Loss(criterion)})


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(X_train)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.iteration % flen(X_train), len(X_train),trainer.state.output))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(X_dev)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, metrics['accuracy'], metrics['nll']))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(X_test)
        metrics = evaluator.state.metrics
        print("Test Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, metrics['accuracy'], metrics['nll']))



    trainer.run(X_train, max_epochs=100)