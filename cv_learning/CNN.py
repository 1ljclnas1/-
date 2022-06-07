import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
num_classes = 2
batch_size = 50
learning_rate = 0.001
attr = "earring"
# 读取csv文件
train_csv_path = '../cv_learning/data/train.csv'
test_csv_path = '../cv_learning/data/test.csv'
train_dataset = pd.read_csv(train_csv_path)
test_dataset = pd.read_csv(test_csv_path)

# 数据格式转换
train_x = []
train_y = train_dataset[attr].values
train_img_path = train_dataset['img_path'].values

test_x = []
test_y = test_dataset[attr].values
test_img_path = test_dataset['img_path'].values

for path in train_img_path:
    img_data = cv2.imread(path)
    x, y = img_data.shape[0:2]
    img_data = cv2.resize(img_data, (200, 200))
    train_x.append(img_data)

for path in test_img_path:
    img_data = cv2.imread(path)
    x, y = img_data.shape[0:2]
    img_data = cv2.resize(img_data, (200, 200))
    test_x.append(img_data)

train_x = np.array(train_x, dtype=np.float64)
train_x = np.swapaxes(train_x, 1, 3)
train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)

test_x = np.array(test_x, dtype=np.float64)
test_x = np.swapaxes(test_x, 1, 3)
test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y)

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(50 * 50 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = [] # 用于保存损失
train_accur = [] # 用于保存精度
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    temp_loss = 0
    i = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.long())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        temp_loss += loss.item()
    train_losses.append(temp_loss / (i + 1))
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        AP = 0
        label_arr = list()
        predict_arr = list()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 统计正确率
            predict_arr.extend(predicted.type(torch.long).numpy())
            label_arr.extend(labels.numpy())
        print('Test Accuracy of the model on test images: {} %'.format(100 * correct / total))
        train_accur.append(100 * correct / total)
# Test the model
label_arr = list()
predict_arr = list() # 保存预测值
with torch.no_grad():
    correct = 0
    total = 0
    AP = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # 统计正确率
        predict_arr.extend(predicted.type(torch.long).numpy())
        label_arr.extend(labels.numpy())
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
C = confusion_matrix(label_arr, predict_arr) # 可将'1'等替换成自己的类别，如'cat'。

plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title(attr, y=1.07, fontsize = 18)
        # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
        # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})

plt.savefig("CNN_"+attr, dpi=300)
plt.show()
# 保存每个epoch的每个属性的正确率
eval_acc_csv = pd.DataFrame(train_accur, index=[i for i in range(num_epochs)]).T
eval_acc_csv.to_csv("./result/" + "CNN_" + attr +"-eval_accuracy" + ".csv")
# 保存训练过程的loss
train_losses_csv = pd.DataFrame(train_losses)
train_losses_csv.to_csv("./result/" + "CNN_" + attr + "-losses" + ".csv")
# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')

