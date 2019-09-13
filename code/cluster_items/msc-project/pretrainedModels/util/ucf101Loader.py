import tensorflow as tf

ucf101_dataset, ucf101_info = tfds.load(name="ucf101", with_info=True)
ucf101_train , ucf101_test = ucf101_dataset["train"], ucf101_dataset["test"]
print(type(ucf101_info))
assert isinstance(ucf101_train, tf.data.Dataset)
assert isinstance(ucf101_test, tf.data.Dataset)

print('number of training examples:', ucf101_info.splits["train"].num_examples)
print('number of test examples:', ucf101_info.splits["test"].num_examples)
print('number of labels:', ucf101_info.features["label"].num_classes)
