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


use_cuda = torch.cuda.is_available()

def scale(x):
    return (x - np.mean(x)) / np.std(x)

def success_rate(loader, do_print = False):
    correct = 0
    total = 0
    for ep in range(epochs):
        for data in loader:
            images, labels = data
            if use_cuda:
                outputs = net(Variable(images.cuda()))
                labels = labels.cuda()
            else:
                outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #print("value:----------")
            #print(predicted)
            correct += (predicted == labels).sum()
    rate =  correct / total
    if do_print:
        print('Accuracy of the network on the %d  images: %d %%' % (total, 
        100 * rate))
        print(confusion_matrix(predicted.cpu().numpy(), labels.cpu().numpy()))

    return rate


def accuracy(out, act):
    _, predicted = torch.max(out.data, 1)
    total = act.size(0)
    correct = (predicted == act).sum()
    #conf = confusion_matrix(act, predicted)
    return correct / total

def create_dataset(n_sample_from_class, n_test_sample_from_class, n_valid_percent, batch_size):

    class_label = {"grass" : 0, "ocean" : 1, "redcarpet" : 2, "road" : 3, "wheatfield" : 4}

    folder_train = os.path.join('data', 'train')
    folder_test = os.path.join('data', 'test')

    # 120*180*3
    r_size = (64, 64)
    #t_size = int(64 * 64 * 3)

    n_valid_sample_from_class = int(n_sample_from_class * n_valid_percent)
    n_train_sample_from_class = n_sample_from_class - n_valid_sample_from_class
    
    X_train = np.zeros((n_train_sample_from_class * 5 ,3 * r_size[0] * r_size[1]))
    Y_train = np.zeros(n_train_sample_from_class * 5, dtype=np.long)
    
    X_valid = np.zeros((n_valid_sample_from_class * 5 ,3 * r_size[0] * r_size[1]))
    Y_valid = np.zeros(n_valid_sample_from_class * 5, dtype=np.long)

    for label, y in class_label.items():
        class_path = os.path.join(folder_train, label)
        all_files = os.listdir(class_path)
        all_im_files = [f for f in all_files if f.endswith('JPEG') and not f.startswith('.')]
        im_files = random.sample(all_im_files, n_sample_from_class)
        
        for count,imfile in enumerate(im_files):
            full_file_path = os.path.join(folder_train, label, imfile)
            im = cv2.imread(full_file_path)
            im = cv2.resize(im, r_size, interpolation = cv2.INTER_CUBIC)    
            if im is None:
                sys.exit()
            if count < n_valid_sample_from_class:
                X_valid[y * n_valid_sample_from_class + count] = scale(im.flatten()) 
                Y_valid[y * n_valid_sample_from_class + count] = y
            else:
                c = count - n_valid_sample_from_class
                #print(c, y)
                #print(y * n_sample_from_class + c)
                X_train[y * n_train_sample_from_class + c] = scale(im.flatten()) 

                Y_train[y * n_train_sample_from_class + c] = y
                

    # train
    
    
    X_test = np.zeros((n_test_sample_from_class * 5 ,3 * r_size[0] * r_size[1]))
    Y_true = np.zeros((n_test_sample_from_class * 5), dtype=np.int)


    for label, y in class_label.items():
        class_path = os.path.join(folder_test, label)
        all_files = os.listdir(class_path)
        all_im_files = [f for f in all_files if f.endswith('JPEG') and not f.startswith('.')]
        im_files = random.sample(all_im_files, n_test_sample_from_class)
       
        for count, imfile in enumerate(im_files):
            full_file_path = os.path.join(folder_test, label, imfile)
            im = cv2.imread(full_file_path)
            im = cv2.resize(im, r_size, interpolation = cv2.INTER_CUBIC)    
            if im is None:
                sys.exit()
            X_test[y * n_test_sample_from_class + count] = scale(im.flatten() )

            Y_true[y * n_test_sample_from_class + count] = y


    #X_test = normalizer.transform(X_test)

    X_train = torch.from_numpy(X_train).float()
    #Y_train = torch.from_numpy(Y_train).float()
    Y_train = torch.LongTensor(Y_train.tolist())

    X_valid = torch.from_numpy(X_valid).float()
    #Y_train = torch.from_numpy(Y_train).float()
    Y_valid = torch.LongTensor(Y_valid.tolist())
    X_test = torch.from_numpy(X_test).float()
    #Y_true = torch.from_numpy(Y_true).float()
    Y_true = torch.LongTensor(Y_true.tolist())

    train = data_utils.TensorDataset(X_train, Y_train)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    valid = data_utils.TensorDataset(X_valid, Y_valid)
    valid_loader = data_utils.DataLoader(valid , shuffle=False)

    test = data_utils.TensorDataset(X_test, Y_true)
    test_loader = data_utils.DataLoader(dataset=test, shuffle=True)

    return (train_loader, test_loader, valid_loader) 


n_sample_from_class = 100
n_test_sample_from_class = 20

epochs = 10
batch_size = 50
learning_rate = 0.0001

n_valid_percent = 0.3

train_loader, test_loader, valid_loader = create_dataset(n_sample_from_class, n_test_sample_from_class, n_valid_percent, batch_size)

r_size = (64, 64)

N0 = 64 * 64 * 3
N1 = 1000
N2 = 300
N3 = 300
Nout = 5

for data in train_loader:
    X_valid, Y_valid = data
    if use_cuda:
        X_valid, Y_valid = Variable(X_valid.cuda()), Variable(Y_valid.cuda())
    else:
        X_valid, Y_valid = Variable(X_valid), Variable(Y_valid)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear( N0, N1, bias=True)
        self.fc2 = nn.Linear( N1, N2, bias=True)
        self.fc3 = nn.Linear( N2, N3, bias=True)
        self.fc4 = nn.Linear( N3, Nout, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

#  Create an instance of this network.
net = Net()

if use_cuda:
    net.cuda()


#  Define the Mean Squared error loss function as the criterion for this network's training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#  Print a summary of the network.  Notice that this only shows the layers
print(net)

t_loss = np.zeros(epochs)

for ep in range(epochs):
    #  Create a random permutation of the indices of the row vectors.
    
    #  Run through each mini-batch
    #t_rate = np.zeros(epochs)
    
    for i, data in enumerate(train_loader, 0):  
        inputs, labels = data 
        
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad() 
    
        #  Run the network on each data instance in the minibatch
        #  and then compute the object function value
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
      
        loss.backward()
        optimizer.step()
        
        #t_score = accuracy(outputs, labels.data)
        #t_rate[i] = t_score
        
    
        
    pred_Y_valid = net(X_valid)
    valid_loss = criterion(pred_Y_valid, Y_valid)
    score = accuracy(pred_Y_valid, Y_valid.data)
    t_loss[ep] = valid_loss.data[0]
    
    
    print ('\nEpoch [%d/%d]\n-------\n Validation :: Loss: %.4f, Accuracy : %.4f' %(ep+1, epochs, valid_loss.data[0], score))
    #print ('\nEpoch [%d/%d]\n-------\n Validation ::  Accuracy : %.4f' %(ep+1, epochs, score))




print('Training success rate: %.2f' % success_rate(train_loader))
print('Test success rate: %.2f' % success_rate(test_loader, True))
