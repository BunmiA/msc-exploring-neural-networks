import tensorflow as tf
import tensorflow_datasets as tfds
from util import formaters as f
from util import myCallbacks as c

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 20 #1000

ucf101_dataset, ucf101_info = tfds.load(name="ucf101", with_info=True)
ucf101_train , ucf101_test = ucf101_dataset["train"], ucf101_dataset["test"]
callbacks = c.highAccCallback()


train_label = ucf101_train.map(f.get_label)
test_label = ucf101_test.map(f.get_label)

train_multi_frame_1 = ucf101_train.map(f.format_videos)
train_multi_frame_1 = train_multi_frame_1.map(lambda x: f.select_frame_from_i_to_T(x,0,4))
train_multi_frame_1 = train_multi_frame_1.map(f.get_video)

test_multi_frame_1 = ucf101_test.map(f.format_videos)
test_multi_frame_1 = test_multi_frame_1.map(lambda x: f.select_frame_from_i_to_T(x,0,4))
test_multi_frame_1 = test_multi_frame_1.map(f.get_video)

train_multi_frame_2 = ucf101_train.map(f.format_videos)
train_multi_frame_2 = train_multi_frame_2.map(lambda x:f.select_frame_from_i_to_T(x,2,6))
train_multi_frame_2 = train_multi_frame_2.map(f.get_video)

test_multi_frame_2 = ucf101_test.map(f.format_videos)
test_multi_frame_2 = test_multi_frame_2.map(lambda x: f.select_frame_from_i_to_T(x,2,6))
test_multi_frame_2 = test_multi_frame_2.map(f.get_video)


train_multi_frame_3 = ucf101_train.map(f.format_videos)
train_multi_frame_3 = train_multi_frame_3.map(lambda x:f.select_frame_from_i_to_T(x,4,8))
train_multi_frame_3 = train_multi_frame_3.map(f.get_video)

test_multi_frame_3 = ucf101_test.map(f.format_videos)
test_multi_frame_3 = test_multi_frame_3.map(lambda x:f.select_frame_from_i_to_T(x,4,8))
test_multi_frame_3 = test_multi_frame_3.map(f.get_video)

train_multi_frame_4 = ucf101_train.map(f.format_videos)
train_multi_frame_4 = train_multi_frame_4.map(lambda x:f.select_frame_from_i_to_T(x,6,10))
train_multi_frame_4 = train_multi_frame_4.map(f.get_video)

test_multi_frame_4 = ucf101_test.map(f.format_videos)
test_multi_frame_4 = test_multi_frame_4.map(lambda x:f.select_frame_from_i_to_T(x,6,10))
test_multi_frame_4 = test_multi_frame_4.map(f.get_video)


train_clip_1 = tf.keras.Input(shape=(4,170, 170, 3), name='clip_1')  # Variable-length sequence of ints
train_clip_2 = tf.keras.Input(shape=(4,170, 170, 3), name='clip_2')  # Variable-length sequence of ints
train_clip_3 = tf.keras.Input(shape=(4,170, 170, 3), name='clip_3')  # Variable-length sequence of ints
train_clip_4 = tf.keras.Input(shape=(4,170, 170, 3), name='clip_4')  # Variable-length sequence of ints


train_clip_1_features = tf.keras.layers.Conv3D(96, (4,11,11), strides=3, activation='relu', input_shape=(4,170, 170, 3))(train_clip_1)
train_clip_1_features = tf.keras.layers.Reshape((54, 54, 96))(train_clip_1_features)
train_clip_1_features = tf.keras.layers.BatchNormalization()(train_clip_1_features)
train_clip_1_features = tf.keras.layers.MaxPooling2D(2, 2)(train_clip_1_features)

train_clip_2_features = tf.keras.layers.Conv3D(96, (4,11,11), strides=3, activation='relu', input_shape=(4,170, 170, 3))(train_clip_2)
train_clip_2_features = tf.keras.layers.Reshape((54, 54,96))(train_clip_2_features)
train_clip_2_features = tf.keras.layers.BatchNormalization()(train_clip_2_features)
train_clip_2_features = tf.keras.layers.MaxPooling2D(2, 2)(train_clip_2_features)

train_clip_3_features = tf.keras.layers.Conv3D(96, (4,11,11), strides=3, activation='relu', input_shape=(4,170, 170, 3))(train_clip_3)
train_clip_3_features = tf.keras.layers.Reshape((54, 54,96))(train_clip_3_features)
train_clip_3_features = tf.keras.layers.BatchNormalization()(train_clip_3_features)
train_clip_3_features = tf.keras.layers.MaxPooling2D(2, 2)(train_clip_3_features)

train_clip_4_features = tf.keras.layers.Conv3D(96, (4,11,11), strides=3, activation='relu', input_shape=(4,170, 170, 3))(train_clip_4)
train_clip_4_features = tf.keras.layers.Reshape((54, 54,96))(train_clip_4_features)
train_clip_4_features = tf.keras.layers.BatchNormalization()(train_clip_4_features)
train_clip_4_features = tf.keras.layers.MaxPooling2D(2, 2)(train_clip_4_features)

# train_clip_1_features = tf.keras.layers.Conv2D(256, (5,5), strides=1, activation='relu')

first_half = tf.keras.layers.concatenate([train_clip_1_features, train_clip_2_features])
second_half = tf.keras.layers.concatenate([train_clip_3_features, train_clip_4_features])


first_half = tf.keras.layers.Conv2D(256, (5,5), strides=1, activation='relu')(first_half)
first_half = tf.keras.layers.BatchNormalization()(first_half)
first_half = tf.keras.layers.MaxPooling2D(2, 2)(first_half)

second_half = tf.keras.layers.Conv2D(256, (5,5), strides=1, activation='relu')(second_half)
second_half = tf.keras.layers.BatchNormalization()(second_half)
second_half = tf.keras.layers.MaxPooling2D(2, 2)(second_half)

combined =  tf.keras.layers.concatenate([first_half, second_half])

combined = tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu')(combined)
combined = tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu')(combined)
combined = tf.keras.layers.Conv2D(256, (3,3), strides=1, activation='relu')(combined)
combined = tf.keras.layers.MaxPooling2D(2,2)(combined)
combined = tf.keras.layers.Flatten()(combined)
combined = tf.keras.layers.Dense(4096, activation='relu')(combined)
combined =  tf.keras.layers.Dense(4096, activation='relu')(combined)
label =  tf.keras.layers.Dense(101, activation='softmax')(combined)

slow_fusion_model_bn = tf.keras.Model(inputs=[train_clip_1,train_clip_2,train_clip_3, train_clip_4],
                                      outputs=label)


d = tf.data.Dataset.zip(({'clip_1': train_multi_frame_1, 'clip_2': train_multi_frame_2, 'clip_3': train_multi_frame_3, 'clip_4': train_multi_frame_4}, train_label))
d_batch = d.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
print(d_batch.take(1))

d_test = tf.data.Dataset.zip(({'clip_1': test_multi_frame_1, 'clip_2': test_multi_frame_2, 'clip_3': test_multi_frame_3, 'clip_4': test_multi_frame_4}, test_label))
d_batch_test = d_test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


print('\n# creating slow fusion model using batch normalization')
slow_fusion_model_bn.summary()

sgd = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0.0005)

slow_fusion_model_bn.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

slow_fusion_model_bn.fit(d_batch,
                            epochs=10,
                            verbose=1)

print('\n# Evaluate on test data')
results = slow_fusion_model_bn.evaluate(d_batch_test)
print('test loss, test acc:', results)
