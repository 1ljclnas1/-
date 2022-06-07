import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
import matplotlib.pyplot as plt
import itertools
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
num_classes = 2 # 分类的种类
batch_size = 50
learning_rate = 0.01
attr = "frontal_face"

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# read file
train_csv_path = '../cv_learning/data/train.csv'
test_csv_path = '../cv_learning/data/test.csv'
train_dataset = pd.read_csv(train_csv_path)
test_dataset = pd.read_csv(test_csv_path)

# dataform shift
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
    img_data = cv2.resize(img_data, (200, 200)) # 图像大小相同

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
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

# Convolutional neural network (two convolutional layers)
# 重写torch 里面的module函数
class VGG16(tnn.Module):
    def __init__(self, n_classes=10):
        super(VGG16, self).__init__()

        self.layer1 = vgg_conv_block([3, 16], [16, 16], [5, 5], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([16, 32], [32, 32], [5, 5], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([32, 64, 64], [64, 64, 64], [5, 5, 5], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([64, 32, 32], [32, 32, 32], [5, 5, 5], [1, 1, 1], 2, 2)
        # self.layer5 = vgg_conv_block([32, 16, 16], [16, 16, 16], [5, 5, 5], [1, 1, 1], 2, 2)

        # FC层
        self.layer6 = vgg_fc_layer(1568, 518)
        self.layer7 = vgg_fc_layer(518, 518)

        self.layer8 = tnn.Linear(518, n_classes)

    # 向前输出的一个过程
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        vgg16_features = self.layer4(out)
        # vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        # print(out.shape)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


model = VGG16(num_classes).to(device)

# Loss and optimizer
criterion = tnn.CrossEntropyLoss()
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
        temp = loss.detach().numpy()
        temp_loss += temp
    train_losses.append( temp_loss/(i+1) )
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
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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
plt.savefig("VGG_"+attr, dpi=300)
plt.show()

# 保存每个epoch的每个属性的正确率
eval_acc_csv = pd.DataFrame(train_accur, index=[i for i in range(num_epochs)]).T
eval_acc_csv.to_csv("./result/" + "VGG_" + attr +"-eval_accuracy" + ".csv")
# 保存训练过程的loss
train_losses_csv = pd.DataFrame(train_losses)
train_losses_csv.to_csv("./result/" + "VGG_" + attr + "-losses" + ".csv")
torch.save(model.state_dict(), 'model.ckpt')

