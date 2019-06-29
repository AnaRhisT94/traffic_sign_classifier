# Load pickled data
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# TODO: Fill this in based on where you saved the training and testing data

# Download the data
#import urllib.request
#print('Beginning file download...')
#url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip'
#urllib.request.urlretrieve(url, './traffic-signs-data.zip')

## Load the data ##
training_file = "./data/train.p"
validation_file ="./data/valid.p" 
testing_file = "./data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# Assining the training features and labels   
# 32x32x3 images    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

## Data exploration ##

signnames = pd.read_csv('./signnames.csv')
signnames.set_index('ClassId',inplace=True)

def get_name_from_label(label):
    # Helper, transofrm a numeric label into the corresponding string
    return signnames.loc[label].SignName

# TODO: Number of training examples
n_train = X_train.shape[0] #34799

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0] #12630

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:] # 32x32x3

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train)) # 43 classes

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

## Step2 : Data visualization ##
## Bar Chart to visualize the dataset

# How many examples to every class
n_classes, n_counts = np.unique(y_train, return_counts = True)

# Distribution of the classes, a chart to see how many examples per class
def plot_distribution_of_classes(x_axis, y_axis, x_label, y_label, width = 0.5, color = 'b'):
    plt.figure(figsize = (20,10))
    plt.ylabel(y_label, fontsize = 20)
    plt.xlabel(x_label, fontsize = 20)
    plt.bar(x_axis, y_axis, width, color = color)
    plt.xticks(x_axis)
    plt.show()

    
plot_distribution_of_classes(n_classes, n_counts, "Classes", "Examples", 0.5, 'b')



def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    
    n_images = len(images)
    
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    
    fig = plt.figure(figsize=(4, 4))
    
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a.grid(False)
        a.axis('off')
        # plot matrices (2d arrays) with gray color
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, cmap='gray')
        a.set_title(title)
    
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

# We select random images and display them in console    
def select_random_images_by_classes(features, labels, n_features):
  
  indexes = []
  _classes = np.unique(labels);
  
  while len(indexes) < len(_classes):
  
    # Get a random index of an image (feature vector)
    index = random.randint(0, n_features-1)
    # Get its class
    _class = labels[index]
    
    # If I found the class in _classes, then append it to -1
    # This will make the selection of indexes unique and random
    # in that way we add all the classes and index of random images from every class
    for i in range(0, len(_classes)):

      if _class == _classes[i]:
        _classes[i] = -1
        indexes.append(index)
        break

  images = []
  titles = []

  for i in range(0, len(indexes)):
    images.append(features[indexes[i]])
    titles.append("class " + str(get_name_from_label(labels[indexes[i]])))

  show_images(images, titles = titles)
  
select_random_images_by_classes(X_train, y_train, n_train)


## Augmentation to images
## Thanks to this github rep. https://github.com/vxy10/ImageAugmentation

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

# Testing an image with augmentation
images = []

for i in range(0, 10):
  images.append(transform_image(X_train[10],10,5,5,brightness=1))

show_images(images)

## Optimize distirbution so classes with less than 1k examples will
## have additional 1k - #_of_examples which will be augmented and added
for _class, count in zip(n_classes, n_counts):
  new_images = []
  new_classes = []
  
  if count < 1000:
    y_train_length = y_train.shape[0]
    index = 0
    
    for i in range(0, 1000-count):
      # search for the index that will will represent the class
      # this index is is for matching y_train and x_train
      while y_train[index] != _class:
        index = random.randint(0, y_train_length-1)
      new_images.append(transform_image(X_train[index],10,5,5,brightness=1))
      new_classes.append(_class)
      
    X_train = np.concatenate((X_train, np.array(new_images)))
    y_train = np.concatenate((y_train, np.array(new_classes)))

# count how many examples per class    
n_classes, n_counts = np.unique(y_train, return_counts=True)

# re-plot the dist chart to see the new dist of the classes
plot_distribution_of_classes(n_classes, n_counts, 'Classes', '# Training Examples', 0.7, 'blue')

# Grayscale the images so the NN performance will be higher
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)

X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)

X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)

# check grayscale images
select_random_images_by_classes(X_train_gray.squeeze(), y_train, n_train)

# According to CS231N http://cs231n.github.io/neural-networks-2/#datapre
# It's very important to subtract the mean of the image
# In that way we'll center the cloud data in the origin
X_train_gray -= np.mean(X_train_gray)

X_test_gray -= np.mean(X_test_gray)

X_train = X_train_gray

X_test = X_test_gray

## Now we'll split the data and then shuffle it with sklearn library
# and its sub-library model_selection
# Note: I decided not to use the provided validation data,
# But rather use the augmented data I created to X_train with 51k examples
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
X_train, y_train = shuffle(X_train, y_train)