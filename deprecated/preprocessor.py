# Reminder add the bash commands to dowload the dataset

import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
import json
import os

from fdl_project.classes.pretrained_model import LRSchedule, get_model

from fdl_project.classes.dataset_loader import DatasetLoader
from fdl_project.classes.utils import (
    load_captions_data,
    make_dataset,
    train_val_split,
    custom_standardization,
)

os.environ["KERAS_BACKEND"] = "tensorflow"

keras.utils.set_random_seed(111)


class Preprocessor:
    def __init__(
        self,
        image_size=(299, 299),
        vocab_size=10000,
        seq_length=25,
        embed_dim=512,
        ff_dim=512,
        batch_size=64,
        epochs=30,
    ):
        keras.utils.set_random_seed(111)

        self.IMAGE_SIZE = image_size
        self.VOCAB_SIZE = vocab_size
        self.SEQ_LENGTH = seq_length
        self.EMBED_DIM = embed_dim
        self.FF_DIM = ff_dim
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.AUTOTUNE = tf.data.AUTOTUNE

    def load_and_prepare_data(self):
        # Load the dataset
        dataset = DatasetLoader().load_data(
            folder_name=os.path.join(os.getcwd(), "coco")
        )
        captions_mapping, text_data = load_captions_data(dataset)

        # Split the dataset into training and validation sets
        train_data, valid_data = train_val_split(captions_mapping)
        print("Number of training samples: ", len(train_data))
        print("Number of validation samples: ", len(valid_data))

        # Vectorizing the text data
        strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        strip_chars = strip_chars.replace("<", "")
        strip_chars = strip_chars.replace(">", "")

        # Building a `tf.data.Dataset` pipeline for training
        train_data = {
            os.path.join("coco/train2014/", key): value
            for key, value in train_data.items()
        }

        for key, value in train_data.items():
            if len(value) > 5:
                train_data[key] = value[:5]

        # Pass the list of images and the list of corresponding captions
        train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
        valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

        return train_dataset, valid_dataset
