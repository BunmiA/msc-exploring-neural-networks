import tensorflow as tf
import myCallbacks as c
import formaters as f
import tensorflow_datasets as tfds


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
# import tensorflow_datasets as tfds

IMG_SIZE = 170
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

callbacks = c.highAccCallback()

callbacks = c.highAccCallback()
base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


prediction_layer = tf.keras.layers.Dense(101, activation='softmax')


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dropout(0.5),
    global_average_layer,
    prediction_layer
])

base_learning_rate = 0.1
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(train_batches,
          epochs=5,verbose=1,
          callbacks=[callbacks] )
#                     validation_da

print('\n# Evaluate on test data')
results = model.evaluate(test_batches)
print('test loss, test acc:', results)

print('\n# Evaluate on test data')
results = model.evaluate(test_batches)
print('test loss, test acc:', results)