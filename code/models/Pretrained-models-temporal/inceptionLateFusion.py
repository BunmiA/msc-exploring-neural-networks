import tensorflow as tf
import myCallbacks as c
import tensorflow_datasets as tfds
import formaters as f
import myCallbacks as c
IMG_SIZE = 170
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 20 #1000

ucf101_dataset, ucf101_info = tfds.load(name="ucf101", with_info=True)
ucf101_train , ucf101_test = ucf101_dataset["train"], ucf101_dataset["test"]
callbacks = c.nintyAccCallback()

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

base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE,
                                                            include_top=False,
                                                            weights='imagenet')

base_model.trainable = False
base_model.summary()


train_clip_start = tf.keras.Input(shape=(170, 170, 3), name='start')  # Variable-length sequence of ints
train_clip_end = tf.keras.Input(shape=(170, 170, 3), name='end')  # Variable-length sequence of ints


start_features = base_model(train_clip_start)
start_features = tf.keras.layers.Dropout(0.5)(start_features)


end_features = base_model(train_clip_end)
end_features = tf.keras.layers.Dropout(0.5)(end_features)


combined_features = tf.keras.layers.concatenate([start_features, end_features])

combined_features = tf.keras.layers.GlobalAveragePooling2D()(combined_features)

label = tf.keras.layers.Dense(101, activation='softmax')(combined_features)


inception_late_fusion_model = tf.keras.Model(inputs=[train_clip_start, train_clip_end],
                                             outputs=label)


d = tf.data.Dataset.zip(({'start': train_start, 'end': train_end},train_start_label))
d_batch = d.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

d_test = tf.data.Dataset.zip(({'start': test_start, 'end': test_end},test_start_label))
d_batch_test = d_test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


print('\n# creating late fusion model using inception pretrained model')
inception_late_fusion_model.summary()

sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0005)

inception_late_fusion_model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

inception_late_fusion_model.fit(d_batch,
                                epochs=5,
                                verbose=1,
                                callbacks=[callbacks])


print('\n# Evaluate on test data')
results = inception_late_fusion_model.evaluate(d_batch_test)
print('test loss, test acc:', results)