# Machine Learning - NeuralNetwork

Implementation of different configurations of neural network to approximate the function y = 0.2x^4 + 2x^3 + 0.1x^2 + 10 where (−1 ≤ x ≤ 1).

30'000 samples

Includes functions that accepts actual target and predicted values to calculate mean square error(MSE), mean absolute error (MAE), root mean
square error (RMSE), and r2 score

Implements multiple test cases using different hyper-parameters and two different structures with Fully Connected Layers (FCL) or "Dense" in Tensorflow

Relu and tanh are used as activation functions, "mse" as loss function and ”adam” as optimizer

Structure 1:
    FCL (12 units ) // first hidden layer
    FCL (8 units ) // second hidden layer
    FCL (4 units ) // last hidden layer
Structure 2:
    FCL (24 units ) // one hidden layer
    
**The follwoing applies in all cases**
  – Data split ratios: 30% training, 20% validation, and 50% test set.
  – Number of epochs: 20
  – Bath size: 12
  – Loss function: MSE
  – Optimizer: Adam

**The Code implements each of the following cases and graphs the actual test data & perdicted data**
Case 1:
  – Data: Use shuffled and unscaled data.
  – NN structure: Structure 1.
  – Activation function: Relu.
Case 2:
  – Data: Use shuffled and unscaled data.
  – NN structure: Structure 2.
  – Activation function: Relu.
Case 3:
  – Data: Use shuffled and unscaled data.
  – NN structure: Structure 1.
  – Activation function: tanh.
Case 4:
  – Data: Use shuffled and scaled data. First, shuffle the data then split
  the data and finally scale the data (both x and y).
  – NN structure: Structure 1.
  – Activation function: Relu.
Case 5:
  – Data: Use shuffled and scaled data. First, shuffle the data then split
  the data and finally scale the data (both x and y).
  – NN structure: Structure 1.
  – Activation function: tanh

**Results:**

<img src="[https://github.com/Khurram-0/MachineLearning_NeuralNetwork/blob/main/case1_plots.jpg](https://github.com/Khurram-0/MachineLearning_NeuralNetwork/blob/main/case1_plots.jpg)"  width="100"/>

![](https://github.com/Khurram-0/MachineLearning_NeuralNetwork/blob/main/case1_plots.jpg | width=50)

![alt text](https://github.com/Khurram-0/MachineLearning_NeuralNetwork/blob/main/case2_plots.jpg | width=50)

![alt text](https://github.com/Khurram-0/MachineLearning_NeuralNetwork/blob/main/case3_plots.jpg | width=50)

![alt text](https://github.com/Khurram-0/MachineLearning_NeuralNetwork/blob/main/case4_plots.jpg | width=50)

![alt text](https://github.com/Khurram-0/MachineLearning_NeuralNetwork/blob/main/case5_plots.jpg | width=50)




Some discussions:

**[–Importance of train, validation, and test data–]**
The training set is used to develop a model for the dataset. While the validation set prevents the 
model from, for example, being over fitted. During training the validation set is used to test how 
well the model performs for datasets outside the training set – the matrices determined here are 
used to improve the model during the training phase. The test set is used after the previous 
phases have been completed to test how well the completed model performs. Thus all three 
data sets are of utmost importance in order to develop a good model.

**[–Importance of shuffling the data–]**
Shuffling is very important in order to prevent datasets (train, validation & test) from each just 
getting a chunk of data in a sequential order. If shuffling is not used before or after splitting, it 
would be similar to cutting off sections of our initial plot and assigning them to either train, 
validation &/or test data sets. In other words, each set would be biased

**[–Number of neurons (units) per layer–]**
As the number of neurons per layer is increased the time it takes to complete the training 
process increases significantly. Increasing the number of neutrons can also lead to over-fitting,
because the model may be learning very small details during the training process.
Neural Networks

**[–Number of layers–]**
Increasing the number of layers can help increase the accuracy of the model. If the number of 
layers is increased to too much the accuracy of the model to decrease – due to overfitting. Thus, 
the number of layers can be increased as long as the performance of the network improves.

**[–Difference between tanh and relu activation functions–]**
One of the differences between relu and tanh is that with the relu activation function the 
training process is completed faster. Additionally, the tanh has steeper derivative, this means 
that the gradient is stronger which can lead to a “vanishing gradient” (this can be seen in Case 3)
once the values become very small – the relu activation function avoids this problem since it 
does not have a flat curve. But the relu function is not differentiable at zero, this means some 
neurons may stop responding to changes in the input or error. To help reduce the likely-hood of 
this problem the horizontal part of relu can be slightly inclined.
In conclusion the best practice seems to be to shuffle, split and then scale the data. In addition, it
seems that for most cases the relu activation function is the best for learning a model. These 
model resulting from the cases with these two seem fit the data the best, without over fitting 
(e.g. Case 4).
