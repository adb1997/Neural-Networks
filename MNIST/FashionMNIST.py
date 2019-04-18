import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

#added a callback to stop training as required 

# class myCallback(tf.keras.callbacks.Callback):
  # def on_epoch_end(self, epoch, logs={}):
    # if(logs.get('acc')>0.6):
      # print("\nReached 60% accuracy so cancelling training!")
      # self.model.stop_training = True


callbacks = myCallback()

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])

model.fit(training_images, training_labels, epochs=15,callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

#Gives a 97% accuracy on the MNIST dataset 

#Can be improved further

