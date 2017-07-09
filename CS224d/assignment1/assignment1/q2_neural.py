import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

### 前向运算
    N = Dx
    # 第一个隐层做内积
    a1 = sigmoid(data.dot(W1) + b1)    
    # 第二个隐层做内积
    a2 = softmax(a1.dot(W2) + b2)

    cost = - np.sum(np.log(a2[labels == 1]))/N

    ### 反向传播

    # Calculate analytic gradient for the cross entropy loss function
    grad_a2 = ( a2 - labels ) / N

    # Backpropagate through the second latent layer
    gradW2 = np.dot( a1.T, grad_a2 )
    gradb2 = np.sum( grad_a2, axis=0, keepdims=True )

    # Backpropagate through the first latent layer
    grad_a1 = np.dot( grad_a2, W2.T ) * sigmoid_grad(a1)

    gradW1 = np.dot( data.T, grad_a1 )
    gradb1 = np.sum( grad_a1, axis=0, keepdims=True )

#    if verbose: # Verbose mode for logging information
#        print ("W1 shape: {}".format( str(W1.shape) ))
#        print ("W1 gradient shape: {}".format( str(gradW1.shape) ))
#        print ("b1 shape: {}".format( str(b1.shape) ))
#        print ("b1 gradient shape: {}".format( str(gradb1.shape) ))




    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
#    raise NotImplementedError
    ### END YOUR CODE
    return

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()