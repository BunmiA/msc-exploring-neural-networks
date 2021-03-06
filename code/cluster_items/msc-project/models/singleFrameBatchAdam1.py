import tensorflow as tf
import tensorflow_datasets as tfds
from util import formaters as f
from util import myCallbacks as c

tf.compat.v1.enable_eager_execution()


ucf101_dataset, ucf101_info = tfds.load(name="ucf101", with_info=True)
ucf101_train , ucf101_test = ucf101_dataset["train"], ucf101_dataset["test"]
print(type(ucf101_info))
assert isinstance(ucf101_train, tf.data.Dataset)
assert isinstance(ucf101_test, tf.data.Dataset)

train = ucf101_train.map(f.format_videos)
train = train.map(f.select_first_frame)
train = train.map(f.convert_to_tuple)

test = ucf101_test.map(f.format_videos)
test = test.map(f.select_first_frame)
test = test.map(f.convert_to_tuple)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 20 #1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11,11), strides=3 , activation='relu', input_shape=(170, 170, 3)),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (5,5), strides=1, activation='relu'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu'),

    tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu'),
    tf.keras.layers.Conv2D(256, (3,3), strides=1, activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(4096, activation='relu'),

    tf.keras.layers.Dense(101, activation='softmax')
])

callbacks = c.nintyAccCallback()

print('\n# creating single frame model using batch normalization')
model.summary()

adam = tf.keras.optimizers.Adam(lr=1, beta_1=0.9, beta_2=0.999, epsilon=10^-8, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_batches,
          epochs=10,
          verbose=1,
          callbacks=[callbacks])

print('\n# Evaluate on test data')
results = model.evaluate(test_batches)
print('test loss, test acc:', results)

