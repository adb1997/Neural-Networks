### **---HAND SIGNS---**

The code is for a plain neural network built on Tensorflow to classify handsigns. 
Architecture of the model is as follows : 

CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

Cost function is minimised using Softmax Cross Entropy with logits (https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)
Accuracy on training set : 94.3 %. Can be improved by increasing number of epochs. (Hardware constraints) 

### **---Residual Networks Hand Signs---**

The same hand signs dataset is trained on a deep 50 layer neural network. 

Identity block: CONV2D -> BATCHNORM -> RELU -> CONV2D -> BATCHNORM + RELU where another CONV2D -> BATCHNORM block skips the first two and gets added at the + 

Full Network: CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
   
### **--- MNIST---**

Some neural network classifiers for the MNIST and Fashion MNIST datasets. Experimented with regular neural network in Tensorflow as well as a Convolutional Neural Network to explore accuracy improvement.

