import numpy as np

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['w1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros( hidden_size)
        self.params['w2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        n, d = X.shape
        if n==0:
            print('n==0')
        loss = None
        w1 ,b1 = self.params['w1'],  self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        x2 = X.dot(w1) + b1
        relu_out = x2.copy()
        relu_out[relu_out<0] = 0
        score = relu_out.dot(w2) + b2
        if y is None:
            return score
        score_shift = score - np.max(score, axis=1).reshape(-1,1)
        score_exp = np.exp(score_shift)
        softmax_out = score_exp/np.sum(score_exp, axis=1).reshape(-1,1)
        loss = -np.sum(np.log(softmax_out[range(n), y]))/n + reg*(np.sum(w1*w1) + np.sum(w2*w2))

        grads = {}
        dscore = softmax_out.copy()
        dscore[range(n), y] -= 1
        dscore = dscore/n
        grads['b2'] = np.sum(dscore, axis=0)
        grads['w2'] = relu_out.T.dot(dscore) + reg*w2
        drelu_out = dscore.dot(w2.T)
        dx2 = drelu_out.copy()
        dx2[x2<0] = 0
        grads['w1'] = X.T.dot(dx2) + reg*w1
        grads['b1'] = np.sum(dx2, axis=0)
        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=5000,
              batch_size=20, verbose=False):

        n, d = X.shape
        iterations_per_epoch = max(n / batch_size, 1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for i in range(num_iters):
            index = np.random.choice(n, batch_size, replace=True)
            x_iter = X[index]
            y_iter = y[index]
            loss, grads = self.loss(x_iter, y_iter, reg)
            self.params['w1'] -= learning_rate * grads['w1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['w2'] -= learning_rate * grads['w2']
            self.params['b2'] -= learning_rate * grads['b2']
            if(i!=0 and i%(n/batch_size)==0):
                learning_rate *= learning_rate_decay
                loss_history.append(loss)
                train_acc_history.append((self.predict(x_iter)==y_iter).mean())
                val_acc_history.append((self.predict(X_val)==y_val).mean())

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        y_pred = None
        n, d = X.shape
        w1 ,b1 = self.params['w1'],  self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        x2 = X.dot(w1) + b1
        relu_out = x2.copy()
        relu_out[relu_out<0] = 0
        score = relu_out.dot(w2) + b2
        y_pred = np.argmax(score, axis=1)
        return y_pred


