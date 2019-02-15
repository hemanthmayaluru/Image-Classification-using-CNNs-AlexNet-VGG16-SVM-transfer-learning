
# coding: utf-8

# ## VGG implementation with SVM

# *Python Modules*

# # Note:
# A lot of work here is derivative. Multiple sources have been referred to come up with the architecture and the solution given here though the task as a whole has not been directly used. I will make an effort to refer to the sources these to the end.

# In[52]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import sklearn.svm
from sklearn.model_selection import train_test_split, KFold
import random

plt.ion() 

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
else:
    print("Using CPU")


# ## Dataloader functions
# ImageFolder loads the data directly from its path. transforms are used to then compose the same into the size needed for vggnet and alexnet. The data is then loaded based on the input size. 

# In[53]:


def data_loader(log,data_dir, TRAIN, TEST,  image_crop_size = 224, mini_batch_size = 1 ):
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, TEST]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1,
            shuffle=True, num_workers=1
        )
        for x in [TRAIN, TEST]
    }
    print("Data loading complete")
    return dataloaders, image_datasets
    
def update_details(log, image_datasets):
    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, TEST]}

    for x in [TRAIN, TEST]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x), file = log)

    print("Classes: ", file = log)
    class_names = image_datasets[TRAIN].classes
    classification_size = len(image_datasets[TRAIN].classes)
    print(image_datasets[TRAIN].classes)
    print(classification_size)
    
    return dataset_sizes, classification_size


# ## Setting up the network
# 
# Some utility function to visualize the dataset and the model's predictions

# In[54]:


def set_up_network(net, freeze_training = True, clip_classifier = True, classification_size = 101):
    if net == 'vgg16':
    # Load the pretrained model from pytorch
        network = models.vgg16(pretrained=True)

        # Freeze training for all layers
        # Newly created modules have require_grad=True by default
        if freeze_training:
            for param in network.features.parameters():
                param.require_grad = False

        if clip_classifier:
            features = list(network.classifier.children())[:-5] # Remove last layer
            network.classifier = nn.Sequential(*features) # Replace the model classifier
    
    elif net == 'alexnet':
        network = models.alexnet(pretrained=True)
        if freeze_training:
            for param in network.features.parameters():
                param.require_grad = False
        
        if clip_classifier:
            features = list(network.classifier.children())[:-4] # Remove last layer
            network.classifier = nn.Sequential(*features) # Replace the model classifier
    if classification_size != 1000 and clip_classifier == False:
        num_features = network.classifier[6].in_features
        features = list(network.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, classification_size)]) # Add our layer with 4 outputs
        network.classifier = nn.Sequential(*features) # Replace the model cla
#     print(network)
    return network


# ## Task 1: Update Features
# This function updates the network output for then being able to update it for SVM layer.

# In[55]:


def get_features( log, ipnet, train_batches = 10, number_of_classes = 10 ):

    imgfeatures = []
    imglabels = []
    if classification_size < number_of_classes:
        number_of_classes = classification_size
        print("Input size smaller at:", classification_size,". Adjusting the class to this number", file = log)
    selected_classes = random.sample(range(0,classification_size), number_of_classes)
    print("The selected classes are: ",selected_classes, file = log)
    for i, data in enumerate(dataloaders[TRAIN]):
        if i % 100 == 0:
            print("\rTraining batch {}/{}".format(i, train_batches), file=log)

        # Use half training dataset
        if i > train_batches:
            break

        inputs, labels = data
        if(labels.numpy() not in selected_classes): 
            continue
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        feature = ipnet(inputs)
#         print("The shape of output is: ", feature.shape)
#         print(labels)
        if use_gpu:
            imgfeatures.append(feature.cpu().detach().numpy().flatten())
            imglabels.append(labels.cpu().detach().numpy())
        else:
            imgfeatures.append(feature.detach().numpy().flatten())
            imglabels.append(labels.detach().numpy())
        del inputs, labels, feature
        torch.cuda.empty_cache()
    print("Features Updated")
    return imgfeatures, imglabels


# # Fit features to SVM and predict output

# In[56]:


def fit_features_to_SVM(log, features, labels, train_batch_size,  K=5  ):

    kf = sklearn.model_selection.KFold(n_splits=K)
    kf.get_n_splits(features)
    scores = []
    features = np.array(features)
    labels = np.array(labels)
    print(features.shape)
    print(labels.shape)

    i=0
    for train, test in kf.split(features):
        i+=1
        model = sklearn.svm.SVC(C=100)#, C=1, gamma=0)
        model.fit(features[train, :], labels[train].ravel())
        s=model.score(features[test, :], labels[test])
        print(i,"/",K,"The score for this classification is: ", s, file = log)
        scores.append(s)
    return np.mean(scores), np.std(scores)

# This is an alternative implementation using the same thing.
def fit_features_to_SVM_new( log,features, labels, train_batch_size, K=5  ):
    features = np.array(features)
    labels = np.array(labels)
    scores = []
    for i in range(K):
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=(1/K), random_state=42)
        model = sklearn.svm.SVC(C=100)#, C=1, gamma=0)
        model.fit(x_train, y_train.ravel())
        s=model.score(x_test, y_test)
        print("The score for this classification is: ", s, file = log)
        scores.append(s)
    return np.mean(scores), np.std(scores)


# ## VGG16 implementation with SVM as a classification layer. (All Updates here)
# This updates the data, sets up the network and classifies using SVM.

# In[57]:


# data_dir_10 = "/home/student/meowth/imgClas/food/class10"  
# data_dir_30 = "/home/student/meowth/imgClas/food/class30"
# # data_dir_10 = "C:\DeepLearning\images\class10"  
# # data_dir_30 = "C:\DeepLearning\images\class10"

# TRAIN = 'train'
# TEST = 'test'
# # Set up the network
# vgg16_nc = set_up_network('vgg16', freeze_training = True)
# if use_gpu:
#     vgg16_nc.cuda() #.cuda() will move everything to the GPU side

# ImageDirectory = [data_dir_10, data_dir_30]
# for i, data_dir in enumerate(ImageDirectory):
#     file = open("VGG16_Task1"+str(i)+".txt", "w")
    
#     # Get Data
#     dataloaders, image_datasets = data_loader(file,data_dir, TRAIN, TEST, 
#                                               image_crop_size = 224, mini_batch_size = 1 )
#     dataset_sizes, classification_size = update_details(file, image_datasets)
    
#     # Update train_batch_size
#     train_batch_size = dataset_sizes[TRAIN]
# #     train_batch_size = 10
#     class_size = classification_size
    
#     # Get the image features for the imagenet trained network.
#     imgfeatures_vgg, imglabels_vgg = get_features(file, vgg16_nc, train_batch_size,
#                                                   number_of_classes = class_size)
#     mean_accuracy, sd = fit_features_to_SVM(file, imgfeatures_vgg,
#                                         imglabels_vgg, train_batch_size, K=5)
#     print("The mean and standard deviation of classification for vgg 16 is: ",
#       mean_accuracy, sd, "for class size: ", class_size, file = file)
#     del dataloaders, image_datasets, imgfeatures_vgg, imglabels_vgg
#     file.close()
# del vgg16_nc


# ## Alexnet implementation with SVM as a classification layer. 
# The batch size and other things can be classified from here.

# In[58]:



# # Set up the network
# alex_net_nc = set_up_network('alexnet', freeze_training = True)
# if use_gpu:
#     alex_net_nc.cuda() #.cuda() will move everything to the GPU side

# ImageDirectory = [data_dir_10, data_dir_30]
# for i, data_dir in enumerate(ImageDirectory):
#     file = open("AlexNet_Task1"+str(i)+".txt", "w")
    
#     # Get Data
#     dataloaders, image_datasets = data_loader(file, data_dir, TRAIN, TEST, 
#                                               image_crop_size = 224, mini_batch_size = 1)
#     dataset_sizes, classification_size = update_details(file, image_datasets)
    
#     # Update train_batch_size
#     train_batch_size = dataset_sizes[TRAIN]
# #     train_batch_size = 10
#     class_size = classification_size
    
#     # Get the image features for the imagenet trained network.
#     imgfeatures_alexn, imglabels_alexn = get_features(file, alex_net_nc, train_batch_size,
#                                                       number_of_classes = class_size)
#     mean_accuracy, sd = fit_features_to_SVM(file, imgfeatures_alexn,
#                                         imglabels_alexn, train_batch_size, K=5)
#     print("The mean and standard deviation of classification for AlexNet is: ",
#       mean_accuracy, sd, "for class size: ", class_size, file = file)
#     del dataloaders, image_datasets, imgfeatures_alexn, imglabels_alexn
#     file.close()
# del alex_net_nc


# ## Task 2: This one trains on top of the existing pre-trained network.

# ## Loss function
# Here, based on whether label smoothing is needed or not, a different loss function is selected.

# In[59]:


def cal_loss(pred, gold, smoothing = False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = nn.CrossEntropyLoss()
    return loss


# ## Training with cross-validation.
# Here, a split of 80% for training and 20% for validation is done for cross validation. It otherwise follows the standard training example given in pytorch site.
# 

# In[60]:


def train_model(log, vgg, criterion, optimizer, scheduler, dataloaders, num_epochs=10, label_smoothing = False):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    K = 5
    train_batches = len(dataloaders[TRAIN])
    train_bat = np.ones((train_batches, 1)) # This is a dummy variable as sklearn changed stuff and didn't do it right.
    val_batches = 0.2*train_batches
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs), file=log)
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
       
        kf = sklearn.model_selection.KFold(n_splits=K)
        kf.get_n_splits(train_bat)

    
        for train, test in kf.split(train_bat):
        
            for i, data in enumerate(dataloaders[TRAIN]):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches / 2), end='')

                # Use half training dataset
                if i >= train_batches / 2:
#                 if i >= 1:
                    break
                
                if i not in train:
                    continue
                
                inputs, labels = data

                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
#                 loss_train += loss.item()
                loss_train += loss.data[0]
                acc_train += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            print()
            # * 2 as we only used half of the dataset
            avg_loss = loss_train * 2 / (dataset_sizes[TRAIN]*0.8)
            avg_acc = acc_train * 2 / (dataset_sizes[TRAIN]*0.8)

            vgg.train(False)
            vgg.eval()

            for i, data in enumerate(dataloaders[TRAIN]):
                if i % 5000 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), file=log)

#                 if i >= 1:
#                     break
                if i not in test:
                    continue
                
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(), requires_grad=True), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)

                optimizer.zero_grad()

                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss_val += loss.data[0]
#                 loss_train += loss.item()
                acc_val += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            avg_loss_val = loss_val / (dataset_sizes[TRAIN]*0.2)
            avg_acc_val = acc_val / (dataset_sizes[TRAIN]*0.2)

            print( file = log)
            print("Epoch {} result: ".format(epoch), file = log)
            print("Avg loss (train): {:.4f}".format(avg_loss), file = log)
            print("Avg acc (train): {:.4f}".format(avg_acc), file = log)
            print("Avg loss (val): {:.4f}".format(avg_loss_val), file = log)
            print("Avg acc (val): {:.4f}".format(avg_acc_val), file = log)
            print('-' * 10)
            print()

            if avg_acc_val > best_acc:
                best_acc = avg_acc_val
                best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print(file = log)
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60), file = log)
    print("Best acc: {:.4f}".format(best_acc), file = log)
    
    vgg.load_state_dict(best_model_wts)
    return vgg


# ## Evaluating Model
# In this step, images from validation is chosen and is used for evaluating the trained model.

# In[61]:


def eval_model(log, vgg, criterion, label_smoothing = False):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), file=log)
#         if i >= 1:
#             break
        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), requires_grad=True), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
#         loss = criterion(outputs, labels, smoothing=label_smoothing)
        loss = criterion(outputs, labels)
#         loss_test += loss.data[0]
        loss_test += loss.item()

        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test / dataset_sizes[TEST]
    
    elapsed_time = time.time() - since
    print(file = log)
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60), file = log)
    print("Avg loss (test): {:.4f}".format(avg_loss), file = log)
    print("Avg acc (test): {:.4f}".format(avg_acc), file = log)
    print('-' * 10, file = log)


# In[62]:


lr_=0.05
momentum_=0.9
def set_up_network_param(net_type ='vgg16', freeze_training = False, clip_classifier = False, classification_size=10):
    net = set_up_network(net_type, freeze_training = False, clip_classifier = False, classification_size=10)
    if use_gpu:
        net.cuda() #.cuda() will move everything to the GPU side
#     criterion = cal_loss
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(net.parameters(), lr=lr_, momentum=momentum_)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return net, criterion, optimizer_ft, exp_lr_scheduler


# ## Transfer learning and evaluating VGG model

# In[ ]:


# data_dir_10 = "/home/student/meowth/imgClas/food/class10"  
# data_dir_30 = "/home/student/meowth/imgClas/food/class30"
# data_dir_100 = "/home/student/meowth/imgClas/food/class100"
data_dir_10 = "C:\DeepLearning\images\class10"  
data_dir_30 = "C:\DeepLearning\images\class10"
data_dir_100 = "C:\DeepLearning\images\class10"
ImageDirectory = [data_dir_10, data_dir_30, data_dir_100 ]

TRAIN = 'train'
TEST = 'test'

Epochs = 1
for i, data_dir in enumerate(ImageDirectory):
    file = open("VGG16_Task2"+str(i)+".txt", "w")
    # Get Data
    dataloaders, image_datasets = data_loader(file, data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 10 )
    dataset_sizes, classification_size = update_details(file, image_datasets)
    
    # Set up the network
    vgg16, criterion, optimizer_ft, exp_lr_scheduler = set_up_network_param('vgg16', 
                         freeze_training = False, 
                         clip_classifier = False, 
                         classification_size=classification_size)

    # training the model
    vgg16 = train_model(file, vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=Epochs)
    
    # Testing the model
    print("Testing the trained model", file = file)
    eval_model(file, vgg16, criterion)
    
    # Save the trained Model
    torch.save(vgg16.state_dict(), "VGG16_v1_task2_size_"+str(classification_size)+".pt")
    del vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, image_datasets
    file.close()


# ## Training and evaluating AlexNet

# In[ ]:


Epochs = 1

for i, data_dir in enumerate(ImageDirectory):
    file = open("AlexNet_Task2"+str(i)+".txt", "w")
    # Get Data
    dataloaders, image_datasets = data_loader(file, data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 10 )
    dataset_sizes, classification_size = update_details(file, image_datasets)
    
    # Set up the network
    alexnet, criterion, optimizer_ft, exp_lr_scheduler = set_up_network_param('alexnet', 
                         freeze_training = False, 
                         clip_classifier = False, 
                         classification_size=classification_size)

    # training the model
    alexnet = train_model(file, alexnet, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=Epochs)
    
    # Testing the model
    print("Testing the trained model", file = file)
    eval_model(file, alexnet, criterion)
    
    # Save the trained Model
    torch.save(alexnet.state_dict(), "ALEXNET_v1_task2_size_"+str(classification_size)+".pt")
    del alexnet, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, image_datasets
    file.close()


# ## Task 3: Using label smoothing regularisation
# The loss function is updated to include smoothing and is as shown here.

# ## VGG16 with label smoothing

# In[ ]:


# Epochs = 10

# for i, data_dir in enumerate(ImageDirectory):
#     file = open("VGG16_Task3"+str(i)+".txt", "w")

#     # Get Data
#     dataloaders, image_datasets = data_loader(file, data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 10 )
#     dataset_sizes, classification_size = update_details(file, image_datasets)
    
#     # Set up the network
#     vgg16, criterion, optimizer_ft, exp_lr_scheduler = set_up_network_param('vgg16', 
#                          freeze_training = False, 
#                          clip_classifier = False, 
#                          classification_size=classification_size)

#     # training the model
#     vgg16 = train_model(file, vgg16, 
#                         criterion, optimizer_ft, 
#                         exp_lr_scheduler, dataloaders,
#                         num_epochs=Epochs, label_smoothing = True)
    
#     # Testing the model
#     print("Testing the trained model", file = file)
#     eval_model(file, vgg16, criterion, label_smoothing = True)
    
#     # Save the trained Model
#     torch.save(vgg16.state_dict(), "VGG16_v1_task3_size_"+str(classification_size)+".pt")
#     del vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, image_datasets
#     file.close()


# ## AlexNet with label smoothing

# In[ ]:


# Epochs = 10

# for i, data_dir in enumerate(ImageDirectory):
#     file = open("AlexNet_Task3"+str(i)+".txt", "w")
#     # Get Data
#     dataloaders, image_datasets = data_loader(file, data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 10 )
#     dataset_sizes, classification_size = update_details(file, image_datasets)
    
#     # Set up the network
#     alexnet, criterion, optimizer_ft, exp_lr_scheduler = set_up_network_param('alexnet', 
#                          freeze_training = False, 
#                          clip_classifier = False, 
#                          classification_size=classification_size)

#     # training the model
#     alexnet = train_model(file, alexnet, criterion, 
#                           optimizer_ft, exp_lr_scheduler,
#                           dataloaders, num_epochs=Epochs,
#                          label_smoothing = True)
    
#     # Testing the model
#     print("Testing the trained model", file = file)
#     eval_model(file, alexnet, criterion, label_smoothing = True)
    
#     # Save the trained Model
#     torch.save(alexnet.state_dict(), "ALEXNET_v1_task3_size_"+str(classification_size)+".pt")
#     del alexnet, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, image_datasets
#     file.close()

