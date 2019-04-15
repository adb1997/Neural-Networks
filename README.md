# Neural-Networks
Dabbling in Deep Learning 

The first code is for a plain neural network built on Tensorflow to classify handsigns. 
Architecture of the model is as follows : 

 CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
 Cost function is minimised using Softmax Cross Entropy with logits (https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)
 
 Accuracy on training set : 94.3 %. Can be improved by increasing number of epochs. (Hardware constraints) 
