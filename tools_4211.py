import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 


def generate_8classes_2D(N,noise = 2):
    """
    Generate 8 Gaussian classes in 2D in arrays data and labels.
    Each class contains N objects.
    """
    from numpy.random import rand as ra
    from sklearn.utils import shuffle
    data = np.empty((1,2))
    labels = np.empty((1))
    for i in range(8):
        newclass = np.random.multivariate_normal([ra()*20,ra()*20], 
            [[ra()*noise,0],[0,ra()*noise]],size=N)
        data = np.vstack((data,newclass))
        labels = np.vstack((labels,np.ones((N,1))*i))
    data = data[1:,:] 
    labels = labels[1:]
    labels = labels.ravel()
    # Shuffle the data and labels
    rp = np.random.permutation(8*N)
    data, labels = data[rp-1,:], labels[rp-1]
    return data, labels

def generate_two_spirals(N1=250,N2=250,noise = 0.05, \
    revolutions = 3,randomise_step = True):

    if randomise_step:
        theta1 = np.sort(np.random.rand(N1)*2*np.pi*revolutions) 
        theta2 = np.sort(np.random.rand(N2)*2*np.pi*revolutions) + np.pi
        r1 = theta1 / (2*np.pi*revolutions)
        r2 = (theta2 - np.pi) / (2*np.pi*revolutions)
    else:
        theta1 = np.linspace(0,2*np.pi*revolutions,N1)
        theta2 = np.linspace(0,2*np.pi*revolutions+np.pi,N2)
        r1 = np.linspace(0,1,N1)
        r2 = np.linspace(0,1,N2)
    
    data1 = np.transpose(np.array([np.sin(theta1)*r1,np.cos(theta1)*r1]))
    data2 = np.transpose(np.array([np.sin(theta2)*r2,np.cos(theta2)*r2]))
    data = np.vstack((data1,data2))
    data = np.array(data) + np.random.randn(N1+N2,2) * noise
    labels = np.vstack((np.zeros((N1,1)),np.ones((N2,1))))
    labels = labels.ravel()

    return data, labels
        

def train_test_split(data,labels,prop):
    """
    Split data and labels into training and testing parts.
    Testing data proportion is given by prop. Returns
    trainind_data, testing_data, training_labels, testing_labels
    Note: the data is given in the original order, not shuffled!
    """
    N = data.shape[0] # number of objects
    Ntest = np.round(prop*N).astype(int)
    index = np.random.permutation(N) # subsample index
    index = np.ravel(index[:]) # flatten
    tsd = data[np.sort(index[:Ntest]),:]
    tsl = labels[np.sort(index[:Ntest])]
    trd = data[np.sort(index[Ntest:]),:]
    trl = labels[np.sort(index[Ntest:])]

    return trd, tsd, trl, tsl


def train_test_ldc(training_data,training_labels,testing_data,testing_labels):
    """
    Trains and tests a linear discriminant classifier.
    Returns the classification error, the predicted labels and the classifier.
    """
    cla = LinearDiscriminantAnalysis()
    # Train the classifier with data and labels
    cla.fit(training_data, training_labels)
    assigned_labels = cla.predict(testing_data)
    testing_error = np.mean(testing_labels != assigned_labels)
    return testing_error, assigned_labels, cla

def train_test_tree(training_data,training_labels,testing_data,testing_labels):
    """
    Trains and tests a decision tree classifier.
    Returns the classification error, the predicted labels and the classifier.
    """
    cla = DecisionTreeClassifier()
    # Train the classifier with data and labels
    cla.fit(training_data, training_labels)
    assigned_labels = cla.predict(testing_data)
    testing_error = np.mean(testing_labels != assigned_labels)
    return testing_error, assigned_labels, cla

def train_test_logistic(training_data,training_labels, \
                        testing_data,testing_labels):
    """
    Trains and tests a decision tree classifier.
    Returns the classification error, the predicted labels and the classifier.
    """
    cla = LogisticRegression()
    # Train the classifier with data and labels
    cla.fit(training_data, training_labels)
    assigned_labels = cla.predict(testing_data)
    testing_error = np.mean(testing_labels != assigned_labels)
    return testing_error, assigned_labels, cla



def train_test_knn(training_data,training_labels, \
                        testing_data,testing_labels, k = 1):
    """
    Trains and tests a decision tree classifier.
    Returns the classification error, the predicted labels and the classifier.
    """
    cla = KNeighborsClassifier(n_neighbors = k)
    # Train the classifier with data and labels
    cla.fit(training_data, training_labels)
    assigned_labels = cla.predict(testing_data)
    testing_error = np.mean(testing_labels != assigned_labels)
    return testing_error, assigned_labels, cla


def som_training(data, k = 5, epochs = 20, alpha = 1, eta = 0.999):
    """
    Trains a SOM neural network of size k-by-k. The learning rate alpha
    starts from the given value and, after each epoch, the value decreases
    as alpha * eta. 
    Returns the SOM weights, the k-by-k Index matrix corresponding to 
    the weights, the k-by-k activation matrix, and the vector with data 
    hits. The data hits vector contains the winning node number for 
    each data point.  
    """
    N = len(data) # number of objects
    Index = np.reshape(range(k*k),(k,k))
    neighbours = [] 
    # neighbourhood
    for i in range(k):
        for j in range(k):
            n = []
            if (i-1 >= 0): 
                n.append(Index[i-1,j])
            if (i+1 < k):
                n.append(Index[i+1,j])
            if (j-1 >= 0): 
                n.append(Index[i,j-1])
            if (j+1 < k):
                n.append(Index[i,j+1])

            neighbours.append(n) 

    weights = np.zeros((k*k,data.shape[1]))
    for kkk in range(epochs*N):
        o = data[np.random.randint(N),:] # random training object
        distances = np.sum((weights-o)**2,axis=1)
        winner = np.argmin(distances)
        weights[winner,:] = weights[winner,:] * alpha + o * (1-alpha)
        nw = neighbours[winner]
        # update the neighbours
        weights[nw,:] = weights[nw,:] * alpha**2 + o * (1-alpha)
        if ~(kkk % N):
            # epoch finished
            alpha = alpha * eta
            # print(alpha)

    # check activation rate
    activations = np.zeros((k*k))
    data_hits = np.zeros((N,1))
    for i in range(N):
        o = data[i,:]
        distances = np.sum((weights-o)**2,axis=1)
        winner = np.argmin(distances)
        activations[winner] +=1
        data_hits[i] = winner

    activations = np.reshape(activations,(k,k))
    return weights, Index, activations, data_hits


def plot_2D_data(data,labels,colours = [],msize = 5):
    """
    Plots a 2D data set. The classes are plotted with the colours in 
    array "colours".
    """
    if np.min(labels) > 0:
        # labels in the classifier were 1,2, ...
        labels = labels - 1
        
    if len(colours)>0:
        plt.scatter(data[:,0],data[:,1],c=colours[labels.astype(int),:], \
            marker="o",s = msize)
    else:
        plt.scatter(data[:,0],data[:,1],c=labels.astype(int), \
            marker="o",s = msize)
    
    plt.xlabel('feature x')
    plt.ylabel('feature y')
    plt.grid(True)
    
def plot_ldc_regions(data,labels,colours):
    """
    Plots the LDC classification regions with the colours in 
    array "colours".
    """

    mi = np.min(data,axis=0)
    ma = np.max(data,axis=0)

    # Generate a meshgrid ------------------------------------------------------
    xx, yy = np.meshgrid(np.linspace(mi[0]-1, ma[0]+1, 200), \
        np.linspace(mi[1]-1, ma[1]+1, 200))
    x = np.reshape(xx,(-1,1)).ravel()
    y = np.reshape(yy,(-1,1)).ravel()
    z = np.transpose(np.vstack((x,y)))
    _,_,cla = train_test_ldc(data,labels,data,labels)
    Z = cla.predict(z) # Label the meshgrid points

    plt.scatter(x,y,c=colours[Z.astype(int),:],marker="o",s = 5)
    plt.xlabel('feature x')
    plt.ylabel('feature y')
    plt.grid(True)


def plot_regions(data,cla,colours):
    """
    Plots the classification regions of trained classifier 
    "cla" with the colours in array "colours".
    """
    mi = np.min(data,axis=0)
    ma = np.max(data,axis=0)

    # Generate a meshgrid ------------------------------------------------------
    xx, yy = np.meshgrid(np.linspace(mi[0]-1, ma[0]+1, 200), \
        np.linspace(mi[1]-1, ma[1]+1, 200))
    x = np.reshape(xx,(-1,1)).ravel()
    y = np.reshape(yy,(-1,1)).ravel()
    z = np.transpose(np.vstack((x,y)))
    Z = cla.predict(z) # Label the meshgrid points
    if np.min(Z) > 0:
        # labels in the classifier were 1,2, ...
        Z = Z - 1
    plt.scatter(x,y,c=colours[Z.astype(int),:],marker="o",s = 5)
    plt.xlabel('feature x')
    plt.ylabel('feature y')
    plt.grid(True)

def plot_evelyn(data_evelyn, labels_evelyn, pltaxes = 0):

    # Colour map for Evelyn
    cm = np.array([
        [0.0235,    0.0235,    0.0235],
        [0.5294,    0.7686,    0.7686],
        [0.9961,    0.8784,    0.8000],
        [0.9608,    0.9961,    0.9961],
        [0.8000,    0.9000,    1.0000],
        [0.5000,    0.5000,         0],
        [0.7000,    0.7000,    0.7000],
        [0.4000,    0.4000,    0.8000]])

    # Create coordinates for the pixels
    s = data_evelyn.shape
    x,y = np.meshgrid(range(s[0]), range(s[1]), indexing='ij')
    z = np.hstack((np.reshape(x,(-1,1)),np.reshape(y,(-1,1))))
    # This makes one data set with two columns. Notice the -1
    # in the reshape command - means "any" :)

    # Prepare the labels
    import numpy.matlib as npml # labels must not be flattened
    if np.ndim(labels_evelyn) == 1:
        labels_evelyn = labels_evelyn.reshape(len(labels_evelyn), -1)
    
    temp = npml.repmat(labels_evelyn,1,s[1]) # labels must not be flattened
    lxy = np.reshape(temp,(-1,1))
    pixel_array = np.reshape(data_evelyn,(-1,1))
    pixel_array[lxy==2] = pixel_array[lxy==2]+4 # apply the labels
    pixel_array = np.ravel(pixel_array) # important! Must flatten!!!

    # Plot
    if pltaxes == 0:
        plt.scatter(z[:,1],s[0]-z[:,0],c = cm[pixel_array,:])
    else:
        pltaxes.scatter(z[:,1],s[0]-z[:,0],c = cm[pixel_array,:])
    pltaxes.axis('Equal')
    pltaxes.axis('Off')
    # plt.show() 
    # This should be suspended so that multiple plots are allowed


def convert_to_date(year,month,day):
    dates_array = np.array([np.datetime64('now') for x in range(len(year))])
    for i in range(len(year)):
        tm = "{}".format(month[i]) # convert to string
        tm = tm.zfill(2) # make sure that it is padded a zero if need be
        td = "{}".format(day[i]) 
        td = td.zfill(2)
        t = "{}-{}-{}".format(year[i],tm,td)
        dates_array[i] = np.datetime64(t)  
    return dates_array
