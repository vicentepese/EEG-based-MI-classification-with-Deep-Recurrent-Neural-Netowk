# Package import
import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax, relu, log_softmax, dropout
from torch.nn import MaxPool2d, Conv2d, ReLU, Softmax, BatchNorm3d, RNN, LSTM
from torch.nn import MaxPool3d, Conv3d
from torch.nn import  GRU, RNNCell, LSTMCell, LSTM, GRUCell, Linear, Dropout2d, CrossEntropyLoss
from torch.autograd import Variable
import os
from random import shuffle
from scipy.io import loadmat, savemat
from time import *
from torch.autograd import Variable
import random
random.seed(2001)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
np.random.seed(300)

# Convert CPU variables to CUDA variables 
use_cuda = torch.cuda.is_available()

def get_variables(x):
    """
    Converts the variable "x" to CUDA variable if it is available
    """
    if use_cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """
    Converts the variable to numpy (useful for cuda variables)
    """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

def compute_conv_dim(im_dim, padding, dilation, kernel_size, stride):
    """
    Computes the output dimension of a 2D-Convolutional Layer
    """
    dim = ((im_dim + 2*padding - dilation*(kernel_size -1) -1)/stride)+1
    
    return int(dim)

# RCNN computation
    
def model_computation(input_features, input_dim, batch_size):
    
    """
    Computes a Recurrent Deep-Convolutional Neural Network 
    """
    class Net(nn.Module):
        def __init__(self, input_features, input_dim):
            super().__init__()
            
            # First Convolutional Block
            self.conv11 = Conv3d(in_channels = input_features, out_channels = 32,
                            kernel_size = (1,3,3), stride = (1,1,1), padding = (0,0,0), bias = True)
            
            self.conv12 = Conv3d(in_channels = 32, out_channels = 32,
                            kernel_size = (1,3,3), stride = (1,1,1), padding = 0, bias = True)
            
            self.conv13 = Conv3d(in_channels = 32, out_channels = 32,
                            kernel_size = (1,3,3), stride = (1,1,1), padding = 0, bias = True)
            
            self.conv14 = Conv3d(in_channels = 32, out_channels = 32,
                            kernel_size = (1,3,3), stride = (1,1,1), padding = 0, bias = True)
            
            self.maxPool1 = MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2),
                                      padding = 0 )
            
            self.batchN1 = BatchNorm3d(num_features = 32)
            
            # Compute output dimensions
            dimconv11 = compute_conv_dim(im_dim = input_dim, padding = 0, 
                                         dilation = 1, kernel_size = 3, stride = 1)
            dimconv12 = compute_conv_dim(im_dim = dimconv11, padding = 0, 
                                         dilation = 1, kernel_size = 3, stride = 1)
            dimconv13 = compute_conv_dim(im_dim = dimconv12, padding = 0, 
                                         dilation = 1, kernel_size = 3, stride = 1)
            dimconv14 = compute_conv_dim(im_dim = dimconv13, padding = 0, 
                                         dilation = 1, kernel_size = 3, stride = 1)
            self.dim_max = compute_conv_dim(im_dim = dimconv14, padding = 0, 
                                         dilation = 1, kernel_size = 2, stride = 2)  
            
             # Second Convolutional Block
            self.conv21 = Conv3d(in_channels = 32, out_channels = 64,
                            kernel_size = (1,3,3), stride = (1,1,1), padding = (0,0,0), bias = True)
            
            self.conv22 = Conv3d(in_channels = 64, out_channels = 64,
                            kernel_size = (1,3,3), stride = (1,1,1), padding = 0, bias = True)
            
            self.maxPool2 = MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2),
                                      padding = 0 )
            
            self.batchN2 = BatchNorm3d(num_features = 64)
            
            # Cocmpute output dimensions:
            dimconv21 = compute_conv_dim(im_dim = self.dim_max, padding = 0, 
                                         dilation = 1, kernel_size = 3, stride = 1)
            dimconv22 = compute_conv_dim(im_dim = dimconv21, padding = 0, 
                                         dilation = 1, kernel_size = 3, stride = 1)
        
            self.dim_max2 = compute_conv_dim(im_dim = dimconv22, padding = 0, 
                                         dilation = 1, kernel_size = 2, stride = 2)  
            
            # Third block
            self.conv31 = Conv3d(in_channels = 64, out_channels = 128,
                            kernel_size = (1,3,3), stride = (1,1,1), padding = (0,0,0), bias = True)
            
            
            self.maxPool3 = MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2),
                                      padding = 0 )
            
            self.batchN3 = BatchNorm3d(num_features = 128)
            
            # Compute output dimensions
            dimconv31 = compute_conv_dim(im_dim = self.dim_max2, padding = 0, 
                                         dilation = 1, kernel_size = 3, stride = 1)
          
            self.dim_max3 = compute_conv_dim(im_dim = dimconv31, padding = 0, 
                                         dilation = 1, kernel_size = 2, stride = 2)  
            
            # LSTM 
            self.LSTM = LSTM(input_size = 10368, hidden_size = 128, num_layers = 1)
            
            
            
            self.linear = Linear(in_features = 3712, out_features = 24, bias = True)
            self.l_out = Linear(in_features = 24, out_features = 3, bias = True)
#            
#            
        def forward(self, x):
            out = {}
            
            # First block
            x = relu(self.conv11(x))
            x = relu(self.conv12(x))
            x = relu(self.conv13(x))
            x = relu(self.conv14(x))
            x = self.maxPool1(x)
            x = self.batchN1(x)
            
            # Second block
            x = relu(self.conv21(x))
            
            x = relu(self.conv22(x))
            
            x = self.maxPool2(x)
            
            x = self.batchN2(x)
            
            # Third block 
            x = relu(self.conv31(x))
            
            x = self.maxPool3(x)
            
            x = self.batchN3(x)
            
            #LSTM 
            x = x.reshape([29,8,128*9**2])
            x, _ = self.LSTM(x)
            x = x.reshape([8,29,128])
            
            # Flatten data
            x = x.view(-1,29*128)
#            # Linear Layer
            x = dropout(self.linear(x), p = 0.5)
            out = dropout(log_softmax(self.l_out(x), dim = 1), p =0.5)
            
            return out
            
    net = Net(input_features, input_dim)
    print(net)
    print(net.dim_max3)
    return(net)
        
def set_optimizer(net, learning_rate = 10e-6):
    """
    Set error function (criterion) and optimizer
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay = 0.2)

        
    return criterion, optimizer

def test_pass(net):
    x = np.random.normal(0,1, (8,3,100,100)).astype("float32")
    out = net(Variable(torch.from_numpy(x)))
    print(out.size())
    print(out)

def data_loader(files, batch_size):
    path = '/home/topocnnbci/Documents/Data/TopomapsMat/'
    
    I = np.ndarray([batch_size,29,3,100,100])
    labels = []
    i = 0
    for file in files: 
        
        # Load the image for the first subject and concatenate the rest of them 
        images = loadmat(path+file)
        targets = images['labels']
        images = images['training_set']
       
        I[i,:,:,:,:] = images
        labels.append(targets)
        i+=1    
        
    I = I.reshape([batch_size,3,29,100,100])
    labels = np.array(labels)
    return I, labels 

def RNN_train (model, criterion, optimizer, batch_size, idxs = 0):
    """
    files - list with the number of files randomly sorted
    batchsize - batchsize
    """
    path = '/home/topocnnbci/Documents/Data/TopomapsMat/'       # Path to the subjects
    files = os.listdir(path)                                    # Subjects list
    
    # Take equal number of classes
    if idxs is 0:
        t = []
        for file in files:
            _ , targets = data_loader([file], 1)
            t.append(targets)
        unique, counts = np.unique(t, return_counts = True)
        min_counts = min(counts)
        
        idxs = list()
        for u in unique:
            idxs.append(np.where(t == u)[0][0:min_counts])
        shuffle(idxs)
        idxs = np.array(idxs)
        idxs = np.reshape(idxs,(idxs.shape[0]*idxs.shape[1]))   
        files = [files[i] for i in idxs]

    else:
        shuffle(idxs)
        files = [files[i] for i in idxs]
        
            
    t_set = files[0:6064]
    v_set = t_set[-606:]
    tst_set = files[6065:]
    
    # Restructure
    idx = 0
    train_set = []
    while idx + batch_size < len(t_set):
        train_set.append(t_set[idx:idx+batch_size])
        idx += batch_size
    
    idx = 0
    val_set = []
    while idx + batch_size < len(v_set):
        val_set.append(v_set[idx:idx+batch_size])
        idx += batch_size
        
    idx = 0
    test_set = []
    while idx + batch_size < len(tst_set):
        test_set.append(tst_set[idx:idx+batch_size])
        idx += batch_size
    
    
#    step = 5 
    epochs = 20
    
    train_accs = []
    train_costs = []
    valid_accs = []
    test_accs = []
    
    for epoch in range(1,epochs +1):
        print('Epoch: ' + str(epoch)+' Training...')
        correct = 0
        # Full pass over training set. The entire pass takes 6 minutes
        model.train()
        for file in train_set:
            # Load files, send to GPU
            inputs, targets = data_loader(file, batch_size)
            targets = targets -1
            targets = torch.from_numpy(targets).to('cuda').long()
            targets = targets.reshape([targets.shape[0]])
            inputs = torch.from_numpy(inputs).to('cuda').float()
            
            # Pass over the network
            optimizer.zero_grad()
            
            out = model(inputs)
                        
            
            batch_loss = criterion(out, targets)
            batch_loss.backward()
            
            optimizer.step()
            train_costs.append(get_numpy(batch_loss))
            preds = np.argmax(get_numpy(out), axis=-1)
            correct += np.sum(get_numpy(targets) == preds)
        
        val_corr = 0
        model.eval()
        for file in val_set:
            # Load files, send to GPU
            inputs, targets = data_loader(file, batch_size)
            targets = targets -1
            targets = torch.from_numpy(targets).to('cuda').long()
            targets = targets.reshape([targets.shape[0]])
            inputs = torch.from_numpy(inputs).to('cuda').float()
            
            out = model(inputs) 
            
            preds_val = np.argmax(get_numpy(out), axis = -1)
            val_corr += np.sum(get_numpy(targets) == preds_val)
            
        test_corr = 0
        for file in test_set:
            # Load files, send to GPU
            inputs, targets = data_loader(file, batch_size)
            targets = targets -1
            targets = torch.from_numpy(targets).to('cuda').long()
            targets = targets.reshape([targets.shape[0]])
            inputs = torch.from_numpy(inputs).to('cuda').float()
            
            out = model(inputs) 
            
            preds_test = np.argmax(get_numpy(out), axis = -1)
            test_corr += np.sum(get_numpy(targets) == preds_test)
        
        test_accs.append(test_corr/(len(test_set)*batch_size))
        valid_accs.append(val_corr/(len(val_set)*batch_size))
        train_accs.append(correct/(batch_size*len(train_set)))
        train_costs.append(np.mean(train_costs))
        print('Training accuracy: ' + str(train_accs[-1]))
        print('Validation accuracy: ' + str(valid_accs[-1]))
        print('Test accuracy: ' + str(test_accs[-1]))
        
    return test_accs, valid_accs, train_accs

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import partial
import os
#from utils import AverageMeter, calculate_accuracy


__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=3, last_fc=True):
        self.last_fc = last_fc

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


         
def set_optimizer(net, learning_rate = 10e-5):
    """
    Set error function (criterion) and optimizer
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
        
    return criterion, optimizer

def test_pass(net):
    x = np.random.normal(0,1, (5,3,32,32)).astype("float32")
    out = net(Variable(torch.from_numpy(x)))
    print(out.size())
    print(out)


def main():
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    model = model_computation(input_features = 3, input_dim = 100, batch_size= batch_size).to(device)
    # remember to change optimizer
    model = resnet200(sample_size = 100, sample_duration = 29).to(device)
    criterion, optimizer = set_optimizer(model)
    idxs = np.load('/home/topocnnbci/Documents/Data/idxs.npy')
    test_accs, valid_accs, train_accs = RNN_train(model, criterion, optimizer, batch_size, idxs)
    
    return  test_accs, valid_accs, train_accs
    

    
test_accs, valid_accs, train_accs  = main()

#t=np.arange(1,21)
#plt.figure(1)
#plt.title('Deep-Recurrent Convolutional Neural Network accuracy', fontsize = 12)
#plt.plot(t,100*train_acc)
#plt.xlabel('Epoch', fontsize = 10)
#plt.xticks(np.arange(0,21,2))
#plt.ylabel('Accuracy (%)', fontsize = 10)
#plt.plot(t, 100*valid_acc)
#plt.plot(t,100*test_acc)
#plt.legend(labels = ('Training', 'Validation', 'Test'))

t=np.arange(1,21)
plt.figure(1)
plt.title('Model Performance Comparison', fontsize = 12)
plt.plot(t,100*train_accCCN ,'b')
plt.xlabel('Epoch', fontsize = 10)
plt.xticks(np.arange(0,21,2))
plt.ylabel('Accuracy (%)', fontsize = 10)
plt.plot(t,100*test_accCNN, 'g')
plt.plot(t, 100*train_accRes, 'b--')
plt.plot(t, 100*test_accRes, 'g--')
plt.legend(labels = ('Training DRCNN', 'Test DRCNN', 'Training ResNet-200', 'Test ResNet-200'))