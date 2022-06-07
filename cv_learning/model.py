import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 4
num_classes = 2 # 分类的种类
batch_size = 50
learning_rate = 0.001

# read file
train_csv_path = '../cv_learning/data/train.csv'
test_csv_path = '../cv_learning/data/test.csv'
train_dataset = pd.read_csv(train_csv_path)
test_dataset = pd.read_csv(test_csv_path)

# dataform shift
train_x = []
train_y = train_dataset['smile'].values
train_img_path = train_dataset['img_path'].values

test_x = []
test_y = test_dataset['smile'].values
test_img_path = test_dataset['img_path'].values
for path in train_img_path:
    img_data = cv2.imread(path)
    x, y = img_data.shape[0:2]
    img_data = cv2.resize(img_data, (224, 224))
    train_x.append(img_data)

for path in test_img_path:
    img_data = cv2.imread(path)
    x, y = img_data.shape[0:2]
    img_data = cv2.resize(img_data, (224, 224)) # 图像大小相同

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
# 重写torch 里面的module函数
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2), #100*100*3 ->
            nn.BatchNorm2d(16),  # guiyihua
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2))
        # 线性分类器输入是25*25*32 num_classes种类的输出
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 48, kernel_size=11),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(48, 128, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(6 * 6 * 256, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(2048, num_classes),
        # )
        # self.fc = nn.Linear(55 * 55 * 32, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(14 * 14 * 32, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(32, num_classes)
        )

    # 向前输出的一个过程
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        # out = self.layer5(out)
        # out = self.features(x)
        # 可以将一个维度为(a,b,c,d)的矩阵转换为一个维度为(b∗c∗d, a)的矩阵
        # out = self.classifier(out)

        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
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
        temp_loss += loss
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
 # 更新混淆矩阵
# def confusion_matrix(preds, labels, conf_matrix):
#     for p, t in zip(preds, labels):
#         conf_matrix[p, t.type(torch.long)] += 1
#     return conf_matrix


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
conf_matrix = torch.zeros(num_classes, num_classes) # 创建一个空矩阵存储混淆矩阵
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
        # conf_matrix = confusion_matrix(predicted, labels=labels, conf_matrix=conf_matrix)
# plot_confusion_matrix(conf_matrix.numpy(), classes=['0', '1'], normalize=False,
#                                  title='Normalized confusion matrix')
C = confusion_matrix(label_arr, predict_arr) # 可将'1'等替换成自己的类别，如'cat'。

plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
# plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
plt.show()


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

