import tensorflow as tf
import tensorflow_datasets as tfds
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pydot
from util import formaters as f
from util import myCallbacks as c

print('\n# Evaluating  on test data')


ucf101_dataset, ucf101_info = tfds.load(name="ucf101", with_info=True)
ucf101_train , ucf101_test = ucf101_dataset["train"], ucf101_dataset["test"]
print(type(ucf101_info))
assert isinstance(ucf101_train, tf.data.Dataset)
assert isinstance(ucf101_test, tf.data.Dataset)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 20 #1000

train_multi_frame_T = ucf101_train.map(f.format_videos)
train_multi_frame_T = train_multi_frame_T.map(lambda x: f.select_frame_from_i_to_T(x,10,20))
train_multi_frame_T = train_multi_frame_T.map(f.convert_to_tuple)

test_multi_frame_T = ucf101_test.map(f.format_videos)
test_multi_frame_T = test_multi_frame_T.map(lambda x:f.select_frame_from_i_to_T(x,10,20))
test_multi_frame_T = test_multi_frame_T.map(f.convert_to_tuple)

train_multi_frame_T_batches = train_multi_frame_T.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_multi_frame_T_batches = test_multi_frame_T.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


model_early_fusion_batch = tf.keras.models.Sequential([
    tf.keras.layers.Conv3D(96, (10,11,11), strides=3 , activation='relu', input_shape=(10,170, 170, 3)),
    tf.keras.layers.Reshape((54, 54,96)),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (5,5), strides=1, activation='relu'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu'),

    tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu'),
    tf.keras.layers.Conv2D(256, (3,3), strides=1, activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(101, activation='softmax')
])

callbacks = c.highAccCallback()

print('\n# creating early fusion model using batch normalization')
model_early_fusion_batch.summary()

sgd = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0.0005)

model_early_fusion_batch.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_early_fusion_batch.fit(train_multi_frame_T_batches,
                           epochs=20,
                           verbose=1,
                           callbacks=[callbacks])

print('\n# Evaluate on test data')
results = model_early_fusion_batch.evaluate(test_multi_frame_T_batches)
print('test loss, test acc:', results)

