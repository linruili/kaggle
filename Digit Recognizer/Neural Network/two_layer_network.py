import numpy as np
from neural_network import TwoLayerNet
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------#
# Create a small net and some toy data to check your implementations.
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()

#----------------------------------------------------------------------#
scores = net.loss(X)
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])

# The difference should be very small. We get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))

#loss
loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))

#----------------------------------------------------------------------#
#train a network
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])
print('Final val accuracy: ', stats['val_acc_history'][-1])

#----------------------------------------------------------------------#
#load data
labeled_images = pd.read_csv("../data/train.csv", nrows=2000)  # type : class 'pandas.core.frame.DataFrame'
labeled_images = labeled_images.fillna(0)
labeled_images[labeled_images>0] = 1
images = labeled_images.iloc[:, 1:]
labels = labeled_images.iloc[:, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,
                                                                              random_state=0)
train_images, valiate_images, train_labels, valiate_labels = train_test_split(train_images, train_labels, train_size=0.8,
                                                                              random_state=0)

train_images = np.array(train_images).reshape(-1,784)
valiate_labels = np.array(valiate_labels)
train_labels = np.array(train_labels)
valiate_images = np.array(valiate_images).reshape(-1,784)
test_labels = np.array(test_labels)
test_images = np.array(test_images).reshape(-1,784)

input_size = train_images.shape[1]
hidden_size = 50
num_classes = 10
num_inputs = train_images.shape[0]
net = TwoLayerNet(input_size, hidden_size, num_classes)
state = net.train(train_images, train_labels, valiate_images, valiate_labels)

print(len(state['loss_history']))
plt.subplot(3,1,1)
plt.plot(state['loss_history'])
plt.subplot(3,1,2)
plt.plot(state['train_acc_history'])
plt.subplot(3,1,3)
plt.plot(state['val_acc_history'])
plt.show()
print('test accuracy:')
print((net.predict(test_images)==test_labels).mean())