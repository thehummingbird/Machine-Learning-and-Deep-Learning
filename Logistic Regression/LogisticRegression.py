import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.02,iter=10000):
        self.lr = lr
        self.iter = iter

    def sigmoid(self, z):
        return (1/(1 + np.exp(-z)))

    def gradient_descent(self,y,x):
        for i in range(self.iter):
            yhat = self.sigmoid(np.dot(self.w.T,x) + self.c)
            dlw = (2/self.m)*np.dot(x, (yhat-y).T)
            dlc = (2/self.m)*np.sum(yhat-y)
            self.w = self.w - self.lr*dlw
            self.c = self.c - self.lr*dlc

    def run(self, y, x):
        self.m = x.shape[1]
        self.n = x.shape[0]
        self.w = np.zeros((self.n,1))
        self.c = 0
        self.gradient_descent(y,x)
        return self.w, self.c

    def predict(self,x):
        yhat = self.sigmoid(np.dot(self.w.T, x) + self.c)
        out = (yhat > 0.5).astype(int)
        return out

def main():
    # nxm (#n features, #m examples)
    x = np.random.rand(1, 10000)*10
    # decision boundary at x=5 here
    y = (x > 5).astype(int)
    algorithm = LogisticRegression()
    algorithm.run(y, x)
    x = np.array([[8.0, 6.8, 3.3, 4.8, 4.85, 5.2]])
    pred = algorithm.predict(x)
    print(pred) # expected output -> 1 1 0 0 0 1

if __name__ == '__main__':
    main()