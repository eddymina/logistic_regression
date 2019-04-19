import pandas as pd 
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, log_loss


def get_iris_X_y(shuffle_instances=True):
    '''
    The Iris dataset is a famous dataset comprising flower characteristics.
    We'll recast as a binary classification task.
    '''
    iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                   header=None)
    y = iris_df.iloc[0:100, 4].values 
    y = np.where(y == 'Iris-setosa', -1, 1)
    y[y<0] = 0 # convert to 1/0 labels
    X = iris_df.iloc[0:100, [0, 2]].values
    if shuffle_instances:
        X, y = shuffle(X, y, random_state=0)
    return X, y

def get_val(X,y, val_p=.1):
  """
  Get randomly shuffles X and y into Training and Validation sets 
  
  """
  sample_size= int(val_p*len(X))
  stacked=np.column_stack((X, y))
  l_stacked=stacked.tolist()
  validation = np.array([l_stacked.pop(random.randrange(len(l_stacked))) for _ in range(sample_size)])
  stacked = np.array(l_stacked)
  X_train= stacked[:,0:2]
  y_train= np.array([int(i) for i in stacked[:,2]])
  val_x= validation[:,0:2]
  val_y= np.array([int(i) for i in validation[:,2]])
  return X_train,y_train,val_x,val_y

def sigmoid(x):
    """
    Params:
    ---
    
    x: nxd numpy array 
    
    Output:
    ---
    Result of Sigmoid "Squishing" Function: Array of vals in rang {0,1}
    """
    return 1/(1+np.exp(-x))

def predict(weights,x):
    """
    Params:
    ---
    weights: 1d array of weights. 
    x: nd numpy array 
    
    Output:
    ---
    nx1 array with the predicted class probabilities 
    Computes dot product and then applies sigmoid function 
    """
    prediction= sigmoid(np.dot(x,weights))
    return prediction

def log_loss(y,prediction):
    """
    Logistic Regression Loss Function Defined 
    
    Params:
    ---
    
    y= actual class (nx1 array)
    prediction= predicted class (nx1 array)
    
    Returns:
    Average of the loss for a entire set (epoch of data)
    """
    return -.5*np.sum(y*np.log(prediction)+(1-y)*np.log(1-prediction))

def loss_plot(val_loss):
    """
    Show Log Loss Plot 
    """
    plt.figure(1)
    x= range(len(val_loss))
    plt.plot(x,val_loss,color='g')
    plt.xlabel('Epoch')
    plt.ylabel('(Validation) Loss')
    plt.title("Loss vs. Epoch")
    plt.draw()
    
def get_acc(weights,X_train,val_x,y_train,val_y):
    
    """
    Get accuracy (of training and Validation sets) using Sklearn 
    
    """
    train_predictions= np.array([int(np.round(v)) for v in predict(weights,X_train)])
    val_predictions=np.array([int(np.round(v)) for v in predict(weights,val_x)])
    
    print('\n Training Accuracy: {}'.format(accuracy_score(train_predictions, y_train)),'| Validation Accuracy: {}\n'.format(accuracy_score(val_predictions, val_y)))

def boundary_plot(X_train,y_train,w_guess):
    """
    Show how the optimized weights form a boundary. 
    
    """
    plt.figure(0)
    plt.scatter(X_train[:,0], X_train[:,1], c= y_train.ravel())
    ax = plt.gca()
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = -(x_vals * w_guess[0])/w_guess[1]
    plt.plot(x_vals, y_vals, '--', c="red")
    plt.title("Boundary Plot boundary @ w[0]*x1 + w[1]*x2 = 0")
    plt.draw()

###### MAIN FUNCTION #####    
def LR_SGD(X, y, epochs, alpha=0.1, print_every=10, val_p=0.1,plot=True):
    '''
    Parameters
    ---
    X: (n x d) matrix comprising n instances with d dimensions/features.
    y: (n x 1) vector comprising labels (0/1)
    epochs: number of epochs to run
    alpha: the learning rate
    val_p: the percentage of data to use as a validation set

    Returns:
    weights, array of epoch log-likelihoods
    '''
    # Isolate the Data into Train and Validation 
    X_train,y_train,val_x,val_y= get_val(X,y,val_p)
    
    
    print ("Given Data:", len(X), "pts | Training Set Size:",len(X_train),'pts | Validation set {}% of Data ('.format(np.round(val_p*100)),len(val_x),')pts.')
    weights= np.random.rand(X.shape[1]) #initialize random weights (no bias taken into account)
    print ("\nRandom Weights Initialized: {} \n".format(weights))
    print ("\n.....// Training //.....\n")
    training_loss=[]
    validation_loss=[]
    for epoch in range(epochs): #loop through all training data
        
        prediction= predict(weights,X_train) #sig(weights*x) gives the class probaility 
        train_loss= log_loss(y_train,prediction)# Loss is defined per epoch as average of the loss per X,y
        val_loss= log_loss(val_y,predict(weights,val_x))
        
        dw=np.dot((prediction - y_train),X_train)
        weights = weights - alpha*(dw/len(y_train))
        validation_loss.append(val_loss)
        
        if epoch % print_every == 0:
            print ("\n For Epoch:", epoch, "With Training Loss:", np.around(train_loss,decimals=6), "| Validation Loss:",np.around(val_loss,decimals=6))

    get_acc(weights,X_train,val_x,y_train,val_y) 
    
    if plot == True:
      boundary_plot(X_train,y_train,weights)
      loss_plot(validation_loss)  
      plt.show()
    
    return weights 

########## RUN FUNCTION #######################
if __name__ == '__main__':
  X,y= get_iris_X_y()
  np_weights = LR_SGD(X, y, epochs=200, alpha=0.1, print_every=50, val_p=.1,plot=False)
  print ("\n ............................ \n")
  print ("Optimized Weights: ", np_weights)