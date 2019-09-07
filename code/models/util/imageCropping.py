import tensorflow as tf


def centerCropAndRezise(dataset):
    image = dataset["video"]
    centerImage = tf.image.central_crop(image,0.5)
    resizedImage = tf.image.resize(centerImage, [200,200])
    randomImage =  tf.image.random_crop(resizedImage,(170,170,3),seed=None,name=None)
    flipImage = tf.image.random_flip_left_right(randomImage, seed =None)
    dataset["video"] = flipImage
    return dataset



