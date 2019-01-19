# Jaemin Lee (aka, J911)
# 2019

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from custom_capture import CustomCapture
from model import Net

batch_size = 30
learning_rate = 0.001
momentum = 0.8

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

class IMGDataset(Dataset):

    def __init__(self, x, y):
        self.len = len(x)
        self.x_data = x
        self.y_data = y
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def get_train_loader(dataset):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

def make_dataset():
    cc = CustomCapture()
    _train_x_data, _train_y_data = cc.capture('training data capture')

    dataset = IMGDataset(_train_x_data, _train_y_data)
    train_loader = get_train_loader(dataset)

    _test_x_data, _testy_data = cc.capture('test data capture')

    dataset = IMGDataset(_test_x_data, _testy_data)
    test_loader = get_train_loader(dataset)

    print("train_set:", len(train_loader.dataset))
    print("test_set:", len(test_loader.dataset))
    
    return (IMGDataset(_train_x_data, _train_y_data), IMGDataset(_test_x_data, _testy_data))

def save_model(path):
    torch.save(model.state_dict(), path)
    
def train(epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {} Batch: {} Loss: {:.6f}'.format(epoch, batch_idx + 1, loss.item()))

def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)

        test_loss +=  nn.CrossEntropyLoss(reduction='sum')(output, target).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    train_dataset, test_dataset = make_dataset()
    train_loader, test_loader = get_train_loader(train_dataset), get_train_loader(test_dataset)

    for epoch in range(1, 10):
        train(epoch, train_loader)
        test(test_loader)

    save_model('./model_save')