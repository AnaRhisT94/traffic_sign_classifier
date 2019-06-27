# Load pickled data
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
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
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

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
