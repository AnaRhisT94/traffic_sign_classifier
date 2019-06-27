# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

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
>>>>>>> 5a02b2a... Step 0: Load the data.
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