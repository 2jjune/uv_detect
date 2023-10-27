import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras import optimizers


vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layers in (vggmodel.layers):
    layers.trainable = False

x = Flatten()(vggmodel.output)

x = Dense(4096, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(4096, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

predictions = Dense(3, activation='softmax', kernel_initializer='random_uniform',
                    bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

model_final = Model(input = vggmodel.input, output = predictions)
rms = optimizers.RMSprop(lr=0.0001, decay=1e-4)
model_final.compile(loss="categorical_crossentropy", optimizer = rms, metrics=["accuracy"])

print(model_final.summary())