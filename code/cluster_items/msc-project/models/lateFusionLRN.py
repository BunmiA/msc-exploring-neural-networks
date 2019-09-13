import tensorflow as tf
import tensorflow_datasets as tfds
from util import formaters as f
from util import lrnNormalization as l
from util import myCallbacks as c

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 20 #1000

ucf101_dataset, ucf101_info = tfds.load(name="ucf101", with_info=True)
ucf101_train , ucf101_test = ucf101_dataset["train"], ucf101_dataset["test"]
callbacks = c.highAccCallback()

train_start = ucf101_train.map(f.format_videos)
train_start = train_start.map(lambda x: f.select_frame_at_T(x,0))
train_start_all = train_start
train_start_label = train_start.map(f.get_label)
train_start = train_start.map(f.get_video)



test_start = ucf101_test.map(f.format_videos)
test_start = test_start.map(lambda x: f.select_frame_at_T(x,0))
test_start_all = test_start
test_start_label = test_start.map(f.get_label)
test_start = test_start.map(f.get_video)


# test_start, test_start_label = test_start.map(convert_to_tuple)


train_end = ucf101_train.map(f.format_videos)
train_end = train_end.map(lambda x: f.select_frame_at_T(x,15))
train_end_all = train_end
train_end_label = train_end.map(f.get_label)
train_end = train_end.map(f.get_video)


# train_end, train_end_label = train_end.map(convert_to_tuple)

test_end = ucf101_test.map(f.format_videos)
test_end = test_end.map(lambda x: f.select_frame_at_T(x,15))
test_end_all = test_end
test_end_label = test_end.map(f.get_label)
test_end = test_end.map(f.get_video)
# test_end, test_end_label = test_end.map(convert_to_tuple)



train_clip_start = tf.keras.Input(shape=(170, 170, 3), name='start')  # Variable-length sequence of ints
train_clip_end = tf.keras.Input(shape=(170, 170, 3), name='end')  # Variable-length sequence of ints

# Embed each word in the title into a 64-dimensional vector
start_features = tf.keras.layers.Conv2D(96, (11,11), strides=3 , activation='relu', input_shape=(170, 170, 3))(train_clip_start)

start_features = l.MyLRNLayer()(start_features)
start_features = tf.keras.layers.MaxPooling2D(2, 2)(start_features)
start_features = tf.keras.layers.Conv2D(256, (5,5), strides=1, activation='relu')(start_features)

start_features = l.MyLRNLayer()(start_features)
start_features = tf.keras.layers.MaxPooling2D(2,2)(start_features)
start_features = tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu')(start_features)

start_features = tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu')(start_features)
start_features = tf.keras.layers.Conv2D(256, (3,3), strides=1, activation='relu')(start_features)
start_features = tf.keras.layers.MaxPooling2D(2,2)(start_features)


# Embed each word in the title into a 64-dimensional vector
end_features = tf.keras.layers.Conv2D(96, (11,11), strides=3 , activation='relu', input_shape=(170, 170, 3))(train_clip_end)

end_features = l.MyLRNLayer()(end_features)
end_features = tf.keras.layers.MaxPooling2D(2, 2)(end_features)
end_features = tf.keras.layers.Conv2D(256, (5,5), strides=1, activation='relu')(end_features)

end_features = l.MyLRNLayer()(end_features)
end_features = tf.keras.layers.MaxPooling2D(2,2)(end_features)
end_features = tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu')(end_features)


end_features = tf.keras.layers.Conv2D(384, (3,3), strides=1, activation='relu')(end_features)
end_features = tf.keras.layers.Conv2D(256, (3,3), strides=1, activation='relu')(end_features)
end_features = tf.keras.layers.MaxPooling2D(2,2)(end_features)

# Merge all available features into a single large vector via concatenation
x = tf.keras.layers.concatenate([start_features, end_features])
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
label = tf.keras.layers.Dense(101, activation='softmax')(x)

# Instantiate an end-to-end model predicting both priority and department
late_fusion_model_LRN = tf.keras.Model(inputs=[train_clip_start, train_clip_end],
                                       outputs=label)


d = tf.data.Dataset.zip(({'start': train_start, 'end': train_end},train_start_label))
d_batch = d.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

d_test = tf.data.Dataset.zip(({'start': test_start, 'end': test_end},test_start_label))
d_batch_test = d_test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


print('\n# creating late fusion model using local response normalization')
late_fusion_model_LRN.summary()

sgd = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0.0005)

late_fusion_model_LRN.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

late_fusion_model_LRN.fit(d_batch,
          epochs=10,
          verbose=1,
          callbacks=[callbacks])

print('\n# Evaluate on test data')
results = late_fusion_model_LRN.evaluate(d_batch_test)
print('test loss, test acc:', results)
