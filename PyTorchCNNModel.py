from __future__ import print_function
#import enum

from glob import glob
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from skimage import io, transform

import torch.nn as nn
import torch.nn.functional as F


#torch check
#x = torch.rand(5,3)
#print(x)

# cuda check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)




# Define Batch Size
batch_size = 12
number_of_labels = 6

# Define Classes
classes = ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')

# Define Tranforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




# Dataset Path List
DATA_PATH_TRAINING_LIST = glob('D:/Data/SceneImage_Intel/archive/seg_train/*/*.jpg')
DATA_PATH_TEST_LIST = glob('D:/Data/SceneImage_Intel/archive/seg_test/*/*.jpg')
DATA_PATH_PRED_LIST = glob('D:/Data/SceneImage_Intel/archive/seg_pred/*.jpg')
print(len(DATA_PATH_TRAINING_LIST))  #14034
print(len(DATA_PATH_TEST_LIST))      # 3000
print(len(DATA_PATH_PRED_LIST))      # 7301

# Define Path List with every same size of images
def getException(data_path_list, image_size):

    invalid = []
        
    for i, path in enumerate(data_path_list):
        img = Image.open(path)
        imgSize = img.size
        if imgSize != image_size:
            invalid.append(i)

    print("total of different size images: ", str(len(invalid)))
        
    return invalid

# Extract the path list from DATA_PATH_LIST
def extract(data_path_list, image_size): 
    pathlist = data_path_list.copy()  
    exceptionidx = getException(data_path_list, image_size)
      
    for i in exceptionidx:        
        pathlist.remove(data_path_list[i])
        
    print("Every different size images all are removed!!!")
    
    return pathlist

TRAINING_DATAPATH_LIST = extract(DATA_PATH_TRAINING_LIST, (150,150))
TEST_DATAPATH_LIST = extract(DATA_PATH_TEST_LIST, (150,150))
PRED_DATAPATH_LIST = extract(DATA_PATH_PRED_LIST, (150,150))



# Prepare the dataset
def get_label(data_path_list):
    label_list = []
    for path in data_path_list:
        path = path.replace('\\', '/')
        label_list.append(path.split('/')[-2])
    return label_list

class SceneDataset(Dataset):

    # image path -> image list
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    # the number of images
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])

trainloader = torch.utils.data.DataLoader(
    SceneDataset(
        TRAINING_DATAPATH_LIST,
        classes,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True
    )
print("The number of images in a training set is: ", len(trainloader)*batch_size)

testloader = torch.utils.data.DataLoader(
    SceneDataset(
        TEST_DATAPATH_LIST,
        classes,
        transform=transform
        ),
    batch_size=batch_size,
    shuffle=True
    )
print("The number of images in a test set is: ", len(testloader)*batch_size)

predloader = torch.utils.data.DataLoader(
    SceneDataset(
        PRED_DATAPATH_LIST,
        classes,
        transform=transform
        ),
    batch_size=batch_size,
    shuffle=False
    )
print("The number of images in a pred set is: ", len(predloader)*batch_size)

# we want to check if the images are loaded well
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):

    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(' '.join("%10s" % classes[labels[j]] for j in range(batch_size)))
imshow(torchvision.utils.make_grid(images))








# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
# Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> Linear.

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)        
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(114264, 6)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        #print("shape1: " + str(output.shape))

        output = self.pool(output)
        #print("shape2: " + str(output.shape))

        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        #print("shape3: " + str(output.shape))

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        #print("shape4: " + str(output.shape))


        return output

# Train the model on the training data.
from torch.autograd import Variable

# Function to save the model
def saveModel(model):
    path = './sceneClassModel.pth'
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(model):

    model.eval()        #https://bluehorn07.github.io/2021/02/27/model-eval-and-train.html
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)                          #########
            total += labels.size(0)                                            ######### How to caculate them? Study What caculation do this code execute for evaluation.
            accuracy += (predicted == labels).sum().item()                     #########

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    
    return accuracy

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):

    # Instantiate a neural network model 
    model = Network()

    # Define your execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model = model.to(device)

    # Define a loss function
    from torch.optim import Adam

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)


    best_accuracy = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(trainloader, 0):

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using images from training set
            outputs = model(images)

            # compute the loss based on model and real labels
            loss = loss_fn(outputs, labels)

            # backpropagate the loss
            loss.backward()

            # adjust parameters based on the caculated gradients
            optimizer.step()

            # Let's print statistics for every 100 images
            running_loss += loss.item()    # extract the loss value
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch+1, i+1, running_loss/100))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy for this epoch when tested over all test images
        accuracy = testAccuracy(model)
        print('For epoch', epoch+1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy


# Function to test the model with a batch of images and show the labels predictions
def testBatch(model):
    # get batch of images from the test DataLoader
    images, labels = next(iter(testloader))

    # show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # Let's see what if the model identifiers the labels of those example
    outputs = model(images)

    # We got the probability for every labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print("Predicted: ", ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))    

    # show all images as one image grid
    imshow(torchvision.utils.make_grid(images))








# Execute the training model!!!
if __name__ == "__main__":

    #Let's build our model
    train(1000)
    print('Finished Training!!')

    ## Test which classes performed well
    # !!! This code is already applied in train model function !!!
    #testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = './sceneClassModel.pth'
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch(model)