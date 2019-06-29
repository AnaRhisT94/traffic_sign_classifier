## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Data Set Summary & Exploration
I used the numpy library to calculate summary statistics of the traffic signs data set. Initial training data charecteristics is shown below:

The size of training set is 34799
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43

### Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. Below images shows the German Traffic Signs and its class. Image plot showing a random image from every class.

![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/Show%20random%20image%20from%20every%20class.JPG)

Then, in the pre-processing step we see the distirbution of the classes and their examples number:
![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/plot_distribution_of_classes.JPG)

We can see that the distirbution of the classes isn't good, so in the next steps we will fix it.

## First of all, Augmentation
I used the following transformation on the images:
* Translate
* Rotate
* Sheering
* Brighness

## Testing of augmentation of a random image:
![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/Augmentation%20technique.JPG)

## And some more augmentation:
![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/More%20augmentation.JPG)

## Distribution optimization
Now we'll optimize the distribution of the classes and add to the dataset augmented images for all the classes that have less than 1000 examples, and we'll get the following optimized distribution
![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/optimize%20distribution.JPG)

## Network Architecture: LeNet

| Layer                                  |     Description                                                                  |
|:--------------------------:|:------------------------------------------------------:|
| Input                                  | 32x32x1 Grayscale image                                          |
| Convolution_1 5x5       | 1x1 stride, VALID padding, outputs 28X28X6                |
| RELU                                  |                                                                                              |
| Max pooling                    | 2x2 stride,  outputs 14x14x6                                    |
| Convolution_2 5x5       | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU                                  |                                                                                              |
| Max pooling                    | 2x2 stride,  outputs 5x5x16                                       |
| Fully connected_0        | Output = 400.                                                                 |
|Dropout                             |Keep probability – 0.5                                                 |
| Fully connected_1        | Output = 120.                                                                 |
| RELU                                  |                                                                                              |
| Fully connected_2        | Output = 84.                                                                   |
| RELU                                  |                                                                                              |
|Dropout                             |Keep probability = 0.5                                                 |
| Fully connected_3        | Output = 43.                                                                   |
 

## Training parameters values that give 97% accuracy:

* Learning Rate = 0.001
* Epoch = 100
* batch Size = 128
* Dropout at FC 0 layer = 0.5
* Dropout at FC 2 layer = 0.5

## Final results of the training
![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/Accuracy%20Results.JPG)

### Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web: ( 8 total )
https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/pre-process%20the%20predicted%20images.JPG

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                                                |     Prediction                                |
|:----------------------------------------------------:|:---------------------------------------------:|
| Right-of-way at the next Intersection                | Right-of-way at the next Intersection         |
| Speed Limit(30km/h)                                  | Speed Limit(30km/h)                           |
| Priority Road                                        | Priority Road                                 |
| Keep Right                                           | Keep Right                                    |
|Turn Left Ahead                                       | Turn Left Ahead                               |
|General Caution                                       | General Caution                               |
|Road work                                             | Speed Limit(70km/h)                                     |
| Stop                                  | Stop                          |

Meaning that I have missed one prediction, so total of 87.5% accuracy on these 8 images:
![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/Accuracyc%2087.5%25.JPG)

Lastly, Figure showing Top 3 Probabilities for detected images.
![alt text](https://github.com/AnaRhisT94/traffic_sign_classifier/blob/master/Images/top%20k%3D3%20guesses.png)
