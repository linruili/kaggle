import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt

def loadTrainData():
    labeled_images = pd.read_csv("../data/train.csv")  #type : class 'pandas.core.frame.DataFrame'
    images = labeled_images.iloc[:,1:]
    labels = labeled_images.iloc[:,:1]
    train_images, valiate_images, train_labels, valiate_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    return train_images, valiate_images, train_labels, valiate_labels

def train_svm(train_images, valiate_images, train_labels, valiate_labels):
    clf = svm.SVC()
    train_images[train_images>0]  = 1
    valiate_images[valiate_images>0] = 1
    clf.fit(train_images, train_labels.values.ravel())
    print(clf.score(valiate_images, valiate_labels)) #0.9429
    return clf

def predict(clf):
    test_images = pd.read_csv('../data/test.csv')
    test_images[test_images>0] = 1
    result = clf.predict(test_images)
    df = pd.DataFrame(result)
    df.index += 1
    df.columns = ['Label']
    df.to_csv('../data/results.csv', index_label='ImageId')

def showOneImage(image):
    img = image.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()



if __name__=='__main__':
    train_images, valiate_images, train_labels, valiate_labels = loadTrainData()
    clf = train_svm(train_images, valiate_images, train_labels, valiate_labels)
    predict(clf)
