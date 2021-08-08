import numpy as np

class LinearRegression:
    def __init__(self,lr=0.01,iter=10000):
        self.lr = lr
        self.iter = iter

    def gradient_descent(self,y,x):
        for i in range(self.iter):
            yhat = np.dot(self.w.T,x) + self.c
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

def main():
    x = np.random.rand(10,10000) #nxm (n = no of features, m = no of examples)
    wtrue = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[11]])
    ctrue = 3
    y = np.dot(wtrue.T,x) + ctrue + (np.random.randn(1,10000)*0.5)
    algorithm = LinearRegression()
    w,c = algorithm.run(y,x)
    print (w,c) # expected true values of w anc c
    
if __name__ == '__main__':
    main()