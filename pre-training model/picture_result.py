import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
predict_csv = pd.read_csv('result/Resnet18-predict.csv')
label_csv = pd.read_csv('result/Resnet18-label.csv')
selected_attrs = ['hair', 'gender', 'earring', 'smile', 'frontal_face', 'style']  # 选择的属性
for attr in selected_attrs:
    C = confusion_matrix(predict_csv[attr], label_csv[attr])
    print(C)
    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

        # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(attr, y=1.07, fontsize = 18)
        # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
        # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})

    plt.savefig("Resnet18_"+attr, dpi=300)
    plt.show()
