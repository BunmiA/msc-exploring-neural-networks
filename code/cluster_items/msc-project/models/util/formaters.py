import tensorflow as tf
import random as r

def format_videos(dataset, IMG_SIZE=170):
    dataset["video"] = tf.cast(dataset["video"], tf.float32)
    dataset["video"] = (dataset["video"]/255)
    dataset["video"] = tf.image.resize(dataset["video"], (IMG_SIZE,IMG_SIZE))
    return dataset


def select_first_frame(dataset):
    dataset["video"] = dataset["video"][0]
    return dataset

def select_frame_from_i_to_T(dataset,i, T):
    dataset["video"] = dataset["video"][i:T]
    return dataset


def select_frame_at_T(dataset, T):
    dataset["video"] = dataset["video"][T]
    return dataset


def convert_to_tuple(dataset):
    x = dataset["video"]
    y = dataset["label"]

    return x,y

def get_label(dataset):
    x = dataset["label"]
    return x

def get_video(dataset):
    x = dataset["video"]
    return x


def select_random_frame(dataset):
    T = r.randint(0, 26)
    dataset["video"] = dataset["video"][T]
    return dataset



def format_normalize_image(dataset):
    dataset["video"] = (dataset["video"]/255)
    return dataset
