import numpy as np
import random

class Network(object):
    def __init__(self):
       self.d1 = Dense(3,3)
       self.d2 = Dense(3,3)
       self.activation = Sigmoid()
       self.act_out = Softmax()
       
    def predict(self, X):
        n, _ = X.shape
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        probability = self.predict_proba(X) 
        res = []
        for item in probability:
            res.append(np.argmax(item)+1)
        return np.array(res)
    
    def predict_proba(self, X):
        return  self.forward(X)
    
    def forward(self, x):
        # 1 layer
        self.x1_ = self.d1.forward(x)
        self.x1_out = self.activation.forward(self.x1_)
       
        # 2 layer 
        self.x2_ = self.d2.forward(self.x1_out)
        self.x2_out = self.act_out.forward(self.x2_)
        
        return self.x2_out
    
    def backward(self, X, err):
        # 2 layer
        x2_d_act = self.act_out.backward(self.x2_out)*err
        x2_d_xw = self.d2.backward(self.x1_out, x2_d_act)
        
        hidden_err = x2_d_act.dot(self.d2.w.T)
        
        # 1 layer
        x1_d_act = self.activation.backward(self.x1_)*hidden_err
        x1_d_xw = self.d2.backward(X, x1_d_act)
        
        # for 2 layer
        self.grad_2 = x2_d_xw
        self.grad_1 = x1_d_xw
          
    
    def grad_descent(self, lr):
        self.d1.w-=lr*self.grad_1
        self.d2.w-=lr*self.grad_2
 
    
    def MSE(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean(np.mean((y_pred - y) ** 2, axis=0))
        return mse
 
        
    def fit(self, X, y, n_epoch, lr):
        n, _ = X.shape
        losses = []
        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        for i in range(n_epoch):
            y_pred = self.forward(X_train)
            
            # calculate gradient
            self.backward(X_train, (y_pred-y)*2)
            
            mse = np.mean(np.mean((y_pred - y) ** 2, axis=0))
            losses.append(mse)
           
            # step
            self.grad_descent(lr)
            
            if i%100==0:
                print(f"epoch {i+1}, MSE: {mse}")
        return losses
    
    

class Dense:
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w  = np.random.uniform(size=(n_in, n_out))
        
    def forward(self, x):
        return np.dot(x, self.w)
    
    def backward(self, x_in, y_out):
        return x_in.T.dot(y_out)
 
    
class Softmax:
    def forward(self, h):
        return np.exp(h) / (sum(np.exp(-h)))
    
    def backward(self, y):
        softmax_val = self.forward(y)
        return softmax_val*(1.-softmax_val)
 
    
class Sigmoid:
    def forward(self, h):
        return 1. / (1 + np.exp(-h))
    
    def backward(self, y):
        return self.forward(y) * (1. - self.forward(y))

