import numpy as np
import cv2
import sys, os
import random
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim


# create dataset

def create_dataset(n_sample_from_class, n_test_sample_from_class, batch_size):

    class_label = {"grass" : 0, "ocean" : 1, "redcarpet" : 2, "road" : 3, "wheatfield" : 4}

    folder_train = 'data/train'
    folder_test = 'data/test'



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

    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    #Y_train = torch.LongTensor(Y_train)

    X_test = torch.from_numpy(X_test).float()
    Y_true = torch.from_numpy(Y_true).float()

    
    train = data_utils.TensorDataset(X_train, Y_train)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    test = data_utils.TensorDataset(X_test, Y_true)
    test_loader = torch.utils.data.DataLoader(dataset=test, 
                                              shuffle=False)

    return (train_loader, test_loader) 



n_sample_from_class = 100
n_test_sample_from_class = 20

epochs = 100
batch_size = 64
learning_rate = 1e-4

train_loader, test_loader = create_dataset(n_sample_from_class, n_test_sample_from_class, batch_size)

t_size = 64 * 64 * 3
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
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  



#  Print a summary of the network.  Notice that this only shows the layers
print(net)


for ep in range(epochs):
    #  Create a random permutation of the indices of the row vectors.
    
    #  Run through each mini-batch
    for i, data in enumerate(train_loader, 0):  
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad() 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Loss: %.4f' %(ep+1, epochs, loss.data[0]))
        


def success_rate(loader):
    correct = 0
    total = 0
    for ep in range(epochs):
        for data in loader:
            images, labels = data
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            _, labels = torch.max(labels, 1)
            #print("value:----------")
            #print(predicted)
            correct += (predicted == labels).sum()
    rate =  correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * rate))

    return rate


print('Training success rate:', success_rate(train_loader))
print('Test success rate:', success_rate(test_loader))

