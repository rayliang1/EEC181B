'''
Code to test shit
Check original code at: http://www.albertauyeung.com/post/python-matrix-factorization/
'''


import numpy as np


class MF():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            # print(prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)


## To run the program
R = np.array([[5,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,5,4],])
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)
#training_process = mf.train()
#print(mf.P)
#print(mf.Q)
#print(mf.full_matrix())



"""
MAIN SHIT
"""
mf.P = np.random.normal(scale=1./mf.K, size=(mf.num_users, mf.K))
mf.Q = np.random.normal(scale=1./mf.K, size=(mf.num_items, mf.K))
        
# Initialize the biases
mf.b_u = np.zeros(mf.num_users)
mf.b_i = np.zeros(mf.num_items)
mf.b = np.mean(mf.R[np.where(mf.R != 0)])
        
# Create a list of training samples
mf.samples = [
    (i, j, mf.R[i, j])
    for i in range(mf.num_users)
    for j in range(mf.num_items)
    if mf.R[i, j] > 0
]
        
# Perform stochastic gradient descent for number of iterations
training_process = []
np.random.shuffle(mf.samples)

# take i = 3, j = 3, r = 4
prediction = mf.get_rating(3, 3)
e = (4 - prediction)
print('e = ')
print(e)
print('\n')

# Update biases
mf.b_u[3] += mf.alpha * (e - mf.beta * mf.b_u[3])
mf.b_i[3] += mf.alpha * (e - mf.beta * mf.b_i[3])

print(mf.b_u)
print('\n')
print(mf.b_i)
print('\n')

print('P :')
print(mf.P)
print('\n')

print('Q :')
print(mf.Q)
print('\n')


P_i = mf.P[3, :][:]
print('P_i = ')
print(P_i)
print('\n')

print('Q[3,:] = ')
print(mf.Q[3,:])
print('\n')
print('P[3,:] = ')
print(mf.P[3,:])
print('\n')
mf.P[3, :] += mf.alpha * (e * mf.Q[3, :] - mf.beta * mf.P[3,:])
print('P[3,:] = ')
print(mf.P[3,:])
print('\n')
print('Q[3,:] = ')
print(mf.Q[3,:])
print('\n')
mf.Q[3, :] += mf.alpha * (e * P_i - mf.beta * mf.Q[3,:])
print('Q[3,:] = ')
print(mf.Q[3,:])
print('\n')
