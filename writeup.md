# **Behavioral Cloning** 
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/right.jpg "Right camera image"
[image2]: ./examples/right_flipped.jpg "Right camera image flipped"
[image3]: ./examples/left.jpg "Left camera image"
[image4]: ./examples/left_flipped.jpg "Left camera image flipped"
[image5]: ./examples/center.jpg "Center camera image"
[image6]: ./examples/center_flipped.jpg "Center camera image flipped"

[Nvidia paper]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py ../model.h5
```
(in current directory)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 72 (model.py lines 96-112).
There are 5 convolutional layers and 4 fully connected layers. The model also includes RELU layers to introduce nonlinearity (code line 107).
The data is normalized in the model using a Keras lambda layer (code line 91). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 102). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a dataset offered by Udacity.

For details about how I created the training data, see the section Creation of the Training Set & Training Process.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model based on famous network by Nvidia ([Nvidia paper]).
It was developed for the same task.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
To avoid the overfitting, I modified the model by adding dropout layer (model.py line 102).
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.
In order to improve the driving behavior in these cases, I tried to change the architecture somehow:
I tried adding convolutional layers and change filter depth. I noticed that decreasing the number of filters in the first layer decresed the quality of driving. Increasing the number of filters remarkably in the final convolutional layer also influenced badly. Too much dropout layers also influence badly. I also tried different values for offset adding to angles measurements, and it influenced the car ability to recover from going too far left or right, and at the same time could lead to too much sinusoidal trajectory.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 96-114) consisted of a convolution neural network with the following layers and layer sizes:
Convolutional layer 1: depth - 24, filter - (5,5), strides - (2,2), activation with 'relu' function
onvolutional layer 2: depth - 36, filter - (5,5), strides - (2,2),activation with 'relu' function
onvolutional layer 3: depth - 48, filter - (5,5), strides - (2,2),activation with 'relu' function
onvolutional layer 4: depth - 64, filter - (3,3), strides - (1,1),activation with 'relu' function
onvolutional layer 5: depth - 64, filter - (3,3), strides - (1,1),activation with 'relu' function
Dropout: keep probability 0.8
Flatten
Dense, output - 100
Activation with 'relu'
Dense, output -  50
Activation with 'relu'
Dense, output -  10
Activation with 'relu'
Dense, output -  1

#### 3. Creation of the Training Set & Training Process

I used dataset offered by Udacity. The data are loaded in line 82 (function load_dataset, lines 22-29).
I took all images, including images from central, left and right cameras. 
I put 20% of the data into a validation set (line 84).
Then I used generator (as it was shown in one of the lessons) to load data and preprocess it on the fly, in batch size portions to feed into my model.It is described in function generator (lines 46-79). It creates batches of dataset contatinig images and angles: center camera images were used with corresponding angles data, while for the left and right camera images I used angle for center camera with offset of +-0.3. It also flippes all images and multiplies corresponding angles to -1. 
So, here it can be seen an augmentation strategy: the number of images becomes 2 times larger due to the flipping; and a recovery from left or right becomes possible due to adding images for left and right cameras with measurements for center camera + constant offset (here it is 0.3).

I also preprocessed data by normalizing (line 91, keras Lambda module) and cropping the images (line 93, keras Cropping2D module) right in the model.

 Examples of initial images for 3 cameras:
![alt text][image1]
![alt text][image3]
![alt text][image5]

Examples of flipped images for 3 cameras:
![alt text][image2]
![alt text][image4]
![alt text][image6]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Video showing the example of the car driving on the base of my model is in current directory (video.mp4).