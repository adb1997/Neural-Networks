"""4 Layer neural network built using Keras and Tensorflow with Relu activation and softmax classifier"""
#Run it to achieve 99.95% accuracy on the training data set and almost close to 100% on the test set

import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks=myCallback()
      
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train =x_train/255
y_train=y_train/255
x_test =x_test/255
y_test=y_test/255

model = tf.keras.models.Sequential([keras.layers.Flatten(input_shape =(28,28)),
                                    keras.layers.Dense(1024, activation=tf.nn.relu),
                                    keras.layers.Dense(512, activation=tf.nn.relu),
                                    keras.layers.Dense(256, activation=tf.nn.relu),
                                    keras.layers.Dense(10, activation=tf.nn.softmax)
                                   ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs =10, callbacks =[callbacks])


model.evaluate(x_test, y_test)