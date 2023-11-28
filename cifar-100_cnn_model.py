import keras.optimizers
import tensorflow as tf
from keras.datasets import cifar100
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#split training test
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#build cnn model

optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=[32, 32, 3]))
cnn.add(tf.keras.layers.Activation(activation="relu"))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Activation(activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.Activation(activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(rate=0.25))
cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.Activation(activation="relu"))
cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.Activation(activation="relu"))
cnn.add(tf.keras.layers.Dropout(rate=0.25))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=256, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=100, activation="softmax"))
cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
cnn.summary()


#train cnn model
history = cnn.fit(X_train, y_train, batch_size=32, epochs=15, verbose=2, validation_split=0.2)

#save model
cnn.save("cnn_model.h5")
print("model saved")
