import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import MNIST

from LeNet import Lenet

def main():
    global lr_scheduler
        #-----------dataset-----------
    train_mnist = MNIST('./data/', train = True, download = True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])) #MNIST\raw
    test_mnist = MNIST('./data/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))  # MNIST\raw
    print(train_mnist.train_data.size())
    print(test_mnist.train_data.size())
    train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=128,
                                                  shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=128,
                                               shuffle=True, num_workers=1)
    #----------network--------
    model = Lenet()
    #---------optimizer\loss\epoches------
    lr = 0.1
    epochs = 100
    CE = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr, momentum=0.9)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    #--------------training---------
    for epoch in range(epochs):
        model.train()
        sum_loss = 0
        for i, (x, label) in enumerate(train_loader):
            # print(x.size())
            optimizer.zero_grad()
            predicts = model(x)
            loss = CE(predicts, label)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        # sum_loss /= train_mnist.train_data.size()[0]
        line = f'{epoch} training loss: {sum_loss}\r\n'
        print(line)
        with open('./result.txt', 'a+', encoding='utf-8') as f:
            f.write(line)
        lr_scheduler.step()
        #----------test-----------  
        if epoch % 5 == 0:
            test_acc = 0.
            model.eval()
            for i, (x, label) in enumerate(test_loader):
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                test_acc += torch.sum(preds == label.data)
            test_acc /= 10000.
            line = f'{epoch} test accuracy: {test_acc}\r\n'
            print(line)
            with open('./result.txt', 'a+', encoding='utf-8') as f:
                f.write(line)



if __name__ == '__main__':
    main()