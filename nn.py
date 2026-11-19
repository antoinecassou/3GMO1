import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

def sigmoid(z):
            return 1/(1+np.exp(-z))


class NN:

    def __init__(
    self,
    X_train, y_train,
    X_test, y_test,
    validations=None,
    number_of_layer=1,
    number_of_hidden_neurals_per_layer=3,
    alpha=1e-4,
    nb_pixels=28
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.validations = validations
        self.number_of_layer = number_of_layer
        self.number_of_hidden_neurals_per_layer = number_of_hidden_neurals_per_layer
        self.alpha = alpha
        self.nb_pixels = nb_pixels

        self.model = None
        self.w_opt = None
        self.b_opt = None


    
    def opt_one_hidden_layer(self,wstar,bstar,x):
        W1 = wstar[0]
        W2 = wstar[1]

        B1 = bstar[0]
        B2 = bstar[1]

        H = sigmoid(np.dot(x,W1)+B1)
        P = sigmoid(H*W2+B2)
        return P

    def opt_two_hidden_layers(self, wstar,bstar,x):
        W1 = wstar[0]
        W2 = wstar[1]
        B1 = bstar[0]
        B2 = bstar[1]

        ax = sigmoid(np.dot(x,W1)+B1)
        H = sigmoid(np.dot(ax,W1)+B1)
        P = sigmoid(np.dot(H,W2)+B2)
        return P
    
    def opt_n_hidden_layers(self, n, wstar,bstar,x):
        W1 = wstar[0]
        W2 = wstar[1]
        B1 = bstar[0]
        B2 = bstar[1]

        for _ in range (n) :
            ax = sigmoid(np.dot(x,W1)+B1)
        H = sigmoid(np.dot(ax,W1)+B1)
        P = sigmoid(np.dot(H,W2)+B2)
        return P

    def objective_hidden_layer(self,w,b):
        P = self.opt_one_hidden_layer(w,b,self.X_train)
        y = self.y_train
        #calcul de la log-loss
        lmbda = self.alpha/len(self.X_train)
        J = lmbda*np.linalg.norm(w)**2 /2 
        L = 0
        for i in range(len(y)):
            p_i = P[i]
            L -= (y[i]*np.log(p_i)+(1-y[i])*np.log(1-p_i))
        L /= len(y)
        J += L

        return J

    def evaluate(self, X, y):
        P = self.opt_one_hidden_layer(self.w_opt, self.b_opt, X)
        return log_loss(y, P)


    def define_weights(self, alpha=None):
        self.model = MLPClassifier(
            hidden_layer_sizes=(h,)*L,
            activation="logistic",
            solver="adam",
            alpha=alpha,
            max_iter=1000,
            random_state=42
            )

        self.model.fit(self.X_train, self.y_train)


    def get_weights(self):
        return self.model.coefs_

    def get_biais(self):
        return self.model.intercepts_

