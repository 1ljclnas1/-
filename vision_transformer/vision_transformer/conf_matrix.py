from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
def two_classes_conf_martix(title=None, cmap=plt.cm.Reds, tp=0, fn=0, fp=0, tn=0):

    plt.rc('font', family='sans-serif', size='4.5')
    plt.rcParams['font.sans-serif'] = ['TP','FP', 'FN', 'TN']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(200,200))
    plt.rcParams['figure.dpi']=200
    cm = np.array([[240,238,50],[158,130,10],[50,0,50]])
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i,j]*100 + 0.5) ==0:
                cm[i,j] = 0
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation = 'nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    plt.title(title, y=1.07, fontsize = 18)
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=list(range(3)), yticklabels=list(range(3)),
            title=title,
            ylabel='True label',
            xlabel='Predicted label'
            )
    
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
            rotation_mode='anchor')
    fmt='d'
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i,j]*100+0.5)>0:
                ax.text(j,i,format(int(cm[i,j]*100+0.5),fmt)+'%',
                        ha="center", va="center",
                        color="white" if cm[i,j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(title+".jpg", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    two_classes_conf_martix("style")
    #two_classes_conf_martix("frontal_face", tp=0, fn=174, fp=0, tn=872)
    #two_classes_conf_martix("hair", tp=0, tn=994, fp=0, fn=52)
    #two_classes_conf_martix("gender", tp=387, tn=624, fp=8, fn=27)
    #two_classes_conf_martix("smile", tp=212, tn=613, fp=57,fn=164)
    #two_classes_conf_martix("earring", tp=859, tn=0, fp=187, fn=0)
