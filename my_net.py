import numpy as np
import cv2
import sys, os
from matplotlib import pyplot as plt
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data as data_utils
import torch.nn.functional as F

# create dataset

class_label = {"grass" : 0, "ocean" : 1, "redcarpet" : 2, "road" : 3, "wheatfield" : 4}

folder_train = 'data/train'
folder_test = 'data/test'

n_sample_from_class = 500
n_test_sample_from_class = 90

# 120*180*3
r_size = (64, 64)
t_size = int(64 * 64 * 3)

X_train = np.ndarray((n_sample_from_class * 5, t_size))
Y_train = np.zeros((n_sample_from_class * 5, 5), dtype=np.uint8)

for label, y in class_label.items():
    class_path = os.path.join(folder_train, label)
    all_files = os.listdir(class_path)
    all_im_files = [f for f in all_files if f.endswith('JPEG')]
    im_files = random.sample(all_im_files, n_sample_from_class)
    
    count = 0
    for imfile in im_files:
        full_file_path = os.path.join(folder_train, label, imfile)
        im = cv2.imread(full_file_path)
        im = cv2.resize(im, r_size, interpolation = cv2.INTER_CUBIC)    
        if im is None:
            sys.exit()
        X_train[y * n_sample_from_class + count] = im.flatten()
        Y_train[y * n_sample_from_class + count, y] = 1
        count += 1

# train
normalizer = preprocessing.StandardScaler().fit(X_train)
X_train = normalizer.transform(X_train)
N = n_sample_from_class * 5
ind_list = [i for i in range(N)]
random.shuffle(ind_list)

X_train = X_train[ind_list]
Y_train = Y_train[ind_list]


X_test = np.ndarray((n_test_sample_from_class * 5, t_size))
Y_true = np.zeros((n_test_sample_from_class * 5, 5), dtype=np.uint8)


for label, y in class_label.items():
    class_path = os.path.join(folder_test, label)
    all_files = os.listdir(class_path)
    all_im_files = [f for f in all_files if f.endswith('JPEG')]
    im_files = random.sample(all_im_files, n_test_sample_from_class)
   
    count = 0
    for imfile in im_files:
        full_file_path = os.path.join(folder_test, label, imfile)
        im = cv2.imread(full_file_path)
        im = cv2.resize(im, r_size, interpolation = cv2.INTER_CUBIC)    
        if im is None:
            sys.exit()
        X_test[y * n_test_sample_from_class + count] = im.flatten()
        Y_true[y * n_test_sample_from_class + count, y] = 1
        count += 1

X_test = normalizer.transform(X_test)

X_train = Variable(torch.from_numpy(X_train).float())
Y_train = Variable(torch.from_numpy(Y_train).float())
#Y_train = torch.LongTensor(Y_train)

X_test = Variable(torch.from_numpy(X_test).float())
Y_true = Variable(torch.from_numpy(Y_true).float())

'''
train = data_utils.TensorDataset(X_train, Y_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

test = data_utils.TensorDataset(X_test, Y_true)
test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False)

'''

N0 = t_size
N1 = 50
N2 = 30
Nout = 5

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear( N0, N1, bias=True)
        self.fc2 = nn.Linear( N1, N2, bias=True)
        self.fc3 = nn.Linear( N2, Nout, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#  Create an instance of this network.
net = Net()

#  Define the Mean Squared error loss function as the criterion for this network's training
criterion = nn.MSELoss()

#  Print a summary of the network.  Notice that this only shows the layers
print(net)



n_train = X_train.data.size()[0]
epochs = 100
batch_size = 10
n_batches = int(np.ceil(n_train / batch_size))
learning_rate = 1e-4

#  Compute an initial loss using all of the validation data.
#
#  A couple of notes are important here:
#  (1) X_valid contains all of the validation input, with each validation
#      data instance being a row of X_valid
#  (2) Therefore, pred_Y_valid is a Variable containing the output layer
#      activations for each of the validation inputs.
#  (3) This is accomplished through the function call net(X_valid), which in
#      turn calls the forward method under the hood to figure out the flow of
#      the data and activations in the network.
#pred_Y_valid = net(X_valid)
#valid_loss = criterion(pred_Y_valid, Y_valid)
#print("Initial loss: %.5f" %valid_loss.data[0])

for ep in range(epochs):
    #  Create a random permutation of the indices of the row vectors.
    indices = torch.randperm(n_train)
    
    #  Run through each mini-batch
    for b in range(n_batches):
        #  Use slicing (of the pytorch Variable) to extract the
        #  indices and then the data instances for the next mini-batch
        batch_indices = indices[b*batch_size:(b+1)*batch_size]
        batch_X = X_train[batch_indices]
        batch_Y = Y_train[batch_indices]
        
        #  Run the network on each data instance in the minibatch
        #  and then compute the object function value
        pred_Y = net(batch_X)
        loss = criterion(pred_Y, batch_Y)
        
        #  Back-propagate the gradient through the network using the
        #  implicitly defined backward function, but zero out the
        #  gradient first.
        net.zero_grad()
        loss.backward()

        #  Complete the mini-batch by actually updating the parameters.
        for param in net.parameters():
            param.data -= learning_rate * param.grad.data
            
    #  Print validation loss every 10 epochs
    #if ep != 0 and ep%10==0:
    #    pred_Y_valid = net(X_valid)
    #    valid_loss = criterion(pred_Y_valid, Y_valid)
    #    print("Epoch %d loss: %.5f" %(ep, valid_loss.data[0]))

#  Compute and print the final training and test loss
#  function values
pred_Y_train = net(X_train)
loss = criterion(pred_Y_train, Y_train)
print('Final training loss is %.5f' %loss.data[0])

pred_Y_test = net(X_test)
test_loss = criterion(pred_Y_test, Y_true)
print("Final test loss: %.5f" %test_loss.data[0])


def success_rate(pred_Y, Y):
    _,pred_Y_index = torch.max(pred_Y, 1)
    _,Y_index = torch.max(Y,1)
    num_equal = torch.sum(pred_Y_index.data == Y_index.data)
    num_different = torch.sum(pred_Y_index.data != Y_index.data)
    rate = num_equal / float(num_equal + num_different)
    return rate

print('Training success rate:', success_rate(pred_Y_train, Y_train))
print('Test success rate:', success_rate(pred_Y_test, Y_true))
