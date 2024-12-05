import numpy as np

# tanh function and its derivative
def tanh(x, deriv=False):
    if(deriv==True):
        return 1 - np.tanh(x)**2
    return np.tanh(x)

# input dataset
X = np.array([ [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# output dataset
y = np.array([[0,0,1,1]]).T

# seed random numbers for reproducibility
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,4)) - 1  # Weights from input layer to hidden layer (4 neurons)
syn1 = 2*np.random.random((4,1)) - 1  # Weights from hidden layer to output layer (1 neuron)

for j in xrange(60000):  # Increased iterations for potentially better learning

    # Forward propagation
    l0 = X
    l1 = tanh(np.dot(l0, syn0))
    l2 = tanh(np.dot(l1, syn1))

    # Calculate the error at the output layer
    l2_error = y - l2

    # Calculate the delta at the output layer
    l2_delta = l2_error * tanh(l2, deriv=True)

    # Backpropagate the error to the hidden layer
    l1_error = l2_delta.dot(syn1.T)

    # Calculate the delta at the hidden layer
    l1_delta = l1_error * tanh(l1, deriv=True)

    # Update the weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print ("Output After Training:")
print (l2)