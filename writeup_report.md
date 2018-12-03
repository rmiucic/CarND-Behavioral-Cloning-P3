# **Behavioral Cloning** 

## Writeup Report

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_anotated.png "Model Visualization"
[image2]: ./examples/center.png "Center"
[image3]: ./examples/recovery.png "Recovery"
[image4]: ./examples/center_fliped.png "Center Flipped"
[image5]: ./examples/left.png "Left"
[image6]: ./examples/left_fliped.png "Left Flipped"
[image7]: ./examples/right.png "Right Flipped"
[image8]: ./examples/right_fliped.png "Right Flipped"
[image9]: ./examples/learning.png "Learning"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 101-119) 

The model includes RELU layers to introduce nonlinearity (e.g. code line 106), and the data is normalized in the model using a Keras lambda layer (code line 103). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 123).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVIDIA network as introduced in the udacity lecture.

My first step was to use a convolution neural network model similar to the LeNet. But this turneout not to work as well as the NVIDIA model. I thought this model might be appropriate because the similar setup was used in the NVIDIA paper. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model did not finish the loop and had worse visual perfomance.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I have recorded more data recovering from the troubled areas. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
I have used keras plot_model module and inkscape for generating the architecture visualization.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the right sides of the road back to center so that the vehicle would learn to overcome the problem of going of the road. Here is an example

![alt text][image3]

I have used center, left right images for the taining. The stering angle for the left/right image was corrected with the following formula: 
```python
correction = 0.2 
steering_left = steering_center + correction
steering_right = steering_center - correction
```

To augment the data set, I flipped images and angles thinking that this would increase the dataset and balance angels For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

After the collection process, I had 7786 number of time instances with 6 images for each. That makes 46716 images for training and validation.

I randomly shuffled the data set and put 10% of the data into a validation set. I have used generators because I had issues with loading all training images into the memory. 

I preprocessed this data by normalizing the image. I used keras Lambda function as a layer in the model with following code:
```python
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
```
Then I croped top 70 pixels of the image to remove the sky/mountains from the image. I have also removed the bottom 25 pixes to remove the hood of the car. I used build in keras method as a layer in the model. 
```python
model.add(Cropping2D(cropping=((70,25), (0,0))))
```

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by learning not improving pass the 5th epoch. 

![alt text][image9]
 I used an adam optimizer so that manually training the learning rate wasn't necessary.
