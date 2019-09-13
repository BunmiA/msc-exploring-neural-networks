
import tensorflow as tf
import tensorflow_datasets as tfds
from util import formaters as f
from util import myCallbacks as c
from util import imageCropping as i

tf.compat.v1.enable_eager_execution()

ucf101_dataset, ucf101_info = tfds.load(name="ucf101", with_info=True)
ucf101_train, ucf101_test = ucf101_dataset["train"], ucf101_dataset["test"]
print(type(ucf101_info))
assert isinstance(ucf101_train, tf.data.Dataset)
assert isinstance(ucf101_test, tf.data.Dataset)

# train = ucf101_train.map(f.format_videos)
train = ucf101_train.map(f.select_first_frame)
train = train.map(i.centerCropAndRezise)
train = train.map(f.convert_to_tuple)
# x_train = train.map(f.get_video)
# y_train = train.map(f.get_label)

# test = ucf101_test.map(f.format_videos)
test = ucf101_train.map(f.select_first_frame)
test = test.map(i.centerCropAndRezise)
test = test.map(f.convert_to_tuple)
# x_test = test.map(f.get_video)
# y_test= test.map(f.get_label)


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 20  # 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11, 11), strides=3, activation='relu', input_shape=(170, 170, 3)),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (5, 5), strides=1, activation='relu'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu'),

    tf.keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(4096, activation='relu'),

    tf.keras.layers.Dense(101, activation='softmax')
])

callbacks = c.nintyAccCallback()

print('\n# creating single frame model using batch normalization')
model.summary()

sgd = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0.0005)

model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:
#
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32, shuffle=True),
#                     steps_per_epoch=len(x_train) / 32, epochs=1, verbose=1,
#                     callbacks=[callbacks])


# print('\n# Evaluate on test data')
# results = model.evaluate_generator(datagen.flow(x_test, y_test, batch_size=32, shuffle=True))
# print('test loss, test acc:', results)

model.fit(train_batches,
          epochs=20,
          verbose=1,
          callbacks=[callbacks])

print('\n# Evaluate on test data')
results = model.evaluate(test_batches)
print('test loss, test acc:', results)
