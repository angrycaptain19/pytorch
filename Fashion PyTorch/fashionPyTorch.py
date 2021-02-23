# imports
# python 3.7.3
# torch 1.7.1
# torchvision .0.8.2
# matplotlib 3.3.3
# numpy 1.19.5
# opencv-python 4.5.1

from time import process_time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

start = process_time()

fout = open('fashiontest.txt', 'w')
transform = transforms.ToTensor()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(device, file=fout)
MODEL_SAVE_PATH = './FashionMNIST_net.pth'
# load datasets
trainset = torchvision.datasets.FashionMNIST(root="./data/FashionMNIST", download=True,
train=False, transform=transform)
testset = torchvision.datasets.FashionMNIST(root="./data/FashionMNIST", download=True,
transform=transform)

BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4
# data loader
trainloader = torch.utils.data.DataLoader(trainset,
    batch_size = BATCH_SIZE_TRAIN, shuffle = True)
testloader = torch.utils.data.DataLoader(testset,
    batch_size = BATCH_SIZE_TEST, shuffle = True)

image, labels = next(iter(trainloader))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ### Inputs to Conv2d: Incoming layers, outgoing layers, Frame size
        # Then Stride, Padding
        self.conv1 = nn.Conv2d(1, 6, 5)

        self.conv2 = nn.Conv2d(6, 12, 5)

        ### Inputs to Linear: Variables/Features in, Variables/Features out
        self.fc1 = nn.Linear(192, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 40)
        self.output = nn.Linear(40, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)      # Activation
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.reshape(-1, 192)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.output(x)
        return x

net = Net()
net.to(device)
learning_rate = .001
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.9)
criterion = nn.CrossEntropyLoss()
TRAIN_EPOCHS = 15
SAVE_EPOCHS = False
SAVE_LAST = False

# Training
for epoch in range(TRAIN_EPOCHS):
    now = process_time()
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds")
    print(f"Beginning Epoch {epoch + 1}...")
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds", file=fout)
    print(f"Beginning Epoch {epoch + 1}...", file=fout, flush=True)
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)\
        # inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:
            # print(f"Epoch: {epoch + 1}, Mini-Batches Processed: {i + 1:5}, Loss: {running_loss/2000:3.5}")
            print(f"Epoch: {epoch + 1}, Mini-Batches Processed: {i + 1:5}, Loss: {running_loss/2000:3.5}", file=fout, flush=True)
            running_loss = 0.0

    now = process_time()
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds")
    print("Starting validation...")
    print(f"[TIMER] Process Time so far: {now - start:.6} seconds", file=fout)
    print("Starting validation...", file=fout, flush=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data
            outputs = net(images)
            # For overall accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[TRAINING] {correct} out of {total}")
    print(f"[TRAINING] {correct} out of {total}", file=fout)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data
            outputs = net(images)
            # For overall accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[VALIDATION] {correct} out of {total}")
    print(f"[VALIDATION] {correct} out of {total}", file=fout, flush=True)
    if SAVE_EPOCHS:
        torch.save(net.state_dict(), f"./saves/FashionMNIST_net_{epoch + 1}.pth")


if TRAIN_EPOCHS:
    print("[INFO] Finished training.")
    print("[INFO] Finished training.", file=fout, flush=True)
    if SAVE_LAST:
        torch.save(net.state_dict(), MODEL_SAVE_PATH)
else:
    net.load_state_dict(torch.load(MODEL_SAVE_PATH))

# testiter = iter(testloader)
# images, labels = testiter.next()
#
# print('Ground Truth:',' '.join(f"{classes[labels[j]]:5}" for j in range(4)))
#
# outputs = net(images)
#
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted:',' '.join(f"{classes[predicted[j]]:5}" for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # images, labels = data
        outputs = net(images)
        # For overall accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # For class-by-class accuracy
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(BATCH_SIZE_TEST):
            label = labels[i]
            try:
                class_correct[label] += c[i].item()
            except:
                class_correct[label] += c.item()
            class_total[label] += 1

print(f"Accuracy of the network on the 10000 test items: {100 * correct / total:.4}%")
print(f"Accuracy of the network on the 10000 test items: {100 * correct / total:.4}%", file=fout)

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
for i in range(10):
    print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.3}%")
    print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.3}%", file=fout)

now = process_time()
print(f"[TIMER] Total Process Time: {now - start:.8} seconds")
print(f"[TIMER] Total Process Time: {now - start:.8} seconds", file=fout, flush=True)
fout.close()
