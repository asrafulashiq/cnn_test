import numpy as np
import cv2
import sys, os
import random
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data as data_utils
import torch.optim as optim

use_cuda = torch.cuda.is_available()

def scale(x):
    return x
    #return (x - np.mean(x)) / np.std(x)

# create dataset

def create_dataset(n_sample_from_class, n_test_sample_from_class, n_valid_percent, batch_size):

    class_label = {"grass" : 0, "ocean" : 1, "redcarpet" : 2, "road" : 3, "wheatfield" : 4}

    folder_train = os.path.join('data', 'train')
    folder_test = os.path.join('data', 'test')

    # 120*180*3
    r_size = (64, 64)
    #t_size = int(64 * 64 * 3)

    n_valid_sample_from_class = int(n_sample_from_class * n_valid_percent)
    n_train_sample_from_class = n_sample_from_class - n_valid_sample_from_class
    
    X_train = np.zeros((n_train_sample_from_class * 5 ,3, r_size[0], r_size[1]))
    Y_train = np.zeros(n_train_sample_from_class * 5, dtype=np.long)
    
    X_valid = np.zeros((n_valid_sample_from_class * 5 ,3, r_size[0], r_size[1]))
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
                X_valid[y * n_valid_sample_from_class + count, 0, :, :] = scale(im[:,:,0]) 
                X_valid[y * n_valid_sample_from_class + count, 1, :, :] = scale(im[:,:,1]) 
                X_valid[y * n_valid_sample_from_class + count, 2, :, :] = scale(im[:,:,2]) 
                Y_valid[y * n_valid_sample_from_class + count] = y
            else:
                c = count - n_valid_sample_from_class
                #print(c, y)
                #print(y * n_sample_from_class + c)
                X_train[y * n_train_sample_from_class + c, 0, :, :] = scale(im[:,:,0]) 
                X_train[y * n_train_sample_from_class + c, 1, :, :] = scale(im[:,:,1]) 
                X_train[y * n_train_sample_from_class + c, 2, :, :] = scale(im[:,:,2]) 
                Y_train[y * n_train_sample_from_class + c] = y
                

    # train
    
    
    X_test = np.zeros((n_test_sample_from_class * 5 ,3, r_size[0], r_size[1]))
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
            X_test[y * n_test_sample_from_class + count, 0, :, :] = scale(im[:,:,0] )
            X_test[y * n_test_sample_from_class + count, 1, :, :] = scale(im[:,:,1] )
            X_test[y * n_test_sample_from_class + count, 2, :, :] = scale(im[:,:,2]) 
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



n_sample_from_class = 500
n_test_sample_from_class = 30

epochs = 100
batch_size = 128
learning_rate = 1e-3

n_valid_percent = 0.3

train_loader, test_loader, valid_loader = create_dataset(n_sample_from_class, n_test_sample_from_class, n_valid_percent, batch_size)

r_size = (64, 64)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 36, kernel_size=5, padding=2),
             nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, padding=2),
             nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(8*8*48, 600)
        self.fc2 = nn.Linear(600, 64)
        self.fc3 = nn.Linear(64, 5)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

#  Create an instance of this network.
net = CNN()

if use_cuda:
    net.cuda()

#  Define the Mean Squared error loss function as the criterion for this network's training
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#  Print a summary of the network.  Notice that this only shows the layers
print(net)

for data in train_loader:
    X_valid, Y_valid = data
    if use_cuda:
        X_valid, Y_valid = Variable(X_valid.cuda()), Variable(Y_valid.cuda())
    else:
        X_valid, Y_valid = Variable(X_valid), Variable(Y_valid)


t_loss = np.zeros(epochs)

for ep in range(epochs):
    #  Create a random permutation of the indices of the row vectors.
    
    #  Run through each mini-batch
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
        
        #  Back-propagate the gradient through the network using the
        #  implicitly defined backward function, but zero out the
        #  gradient first.
        loss.backward()
        optimizer.step()
        
    pred_Y_valid = net(X_valid)
    valid_loss = criterion(pred_Y_valid, Y_valid)
    print ('Epoch [%d/%d], Loss: %.4f' %(ep+1, epochs, valid_loss.data[0]))
    t_loss[ep] = valid_loss.data[0]

#  Compute and print the final training and test loss
#  function values
'''
pred_Y_train = net(X_train)
loss = criterion(pred_Y_train, Y_train)
print('Final training loss is %.5f' %loss.data[0])

pred_Y_test = net(X_test)
test_loss = criterion(pred_Y_test, Y_true)
print("Final test loss: %.5f" %test_loss.data[0])
'''

def success_rate(loader):
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
    print('Accuracy of the network on the %d  images: %d %%' % (total, 
        100 * rate))

    return rate


print('Training success rate:', success_rate(train_loader))
print('Test success rate:', success_rate(test_loader))


