#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_hub as hub

def img_processing(train_dir, IMG_SIZE, valid_split, test_dir=[]):
    train_data = image_dataset_from_directory(directory=train_dir,
                                            label_mode='categorical',
                                            validation_split=valid_split,
                                            subset="training",
                                            seed=42,
                                            image_size=IMG_SIZE)

    valid_data = image_dataset_from_directory(directory=train_dir,
                                            label_mode='categorical',
                                            validation_split=valid_split,
                                            subset="validation",
                                            seed=42,
                                            image_size=IMG_SIZE)

    if len(test_dir)!=0:
        test_data = image_dataset_from_directory(directory=test_dir,
                                            label_mode='categorical',
                                            validation_split=valid_split,
                                            subset="training",
                                            seed=42,
                                            image_size=IMG_SIZE)
        return train_data, valid_data, test_data

    return train_data, valid_data


def feature_create_model(model_url, num_classes):
    feature_extractor_layer = hub.KerasLayer(model_url, 
                                           trainable=False, 
                                           input_shape=(224,224,3),
                                           name="feature_extractor_layer")
    model = tf.keras.Sequential([
      feature_extractor_layer,
      tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model

def main(save_dir, train_dir):
    
    image_size = (200, 200)
    validation_split = 0.2
    num_class = 29
    resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

    train_df, valid_df = img_processing(train_dir, image_size, validation_split)

    model = feature_create_model(resnet_url, num_class)

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(train_df, validation_data=valid_df, epochs=10)

    model.save(save_dir)

if __name__ == "__main__":
    data_path = "/opt/ml/input/data"
    model_dir = "/opt/ml/model"
    train_csv = "train/training_data.csv"
    train_path = f"{data_path}/{train_csv}"
    val_csv = "validation/validation_data.csv"
    val_path = f"{data_path}/{val_csv}"
    
    main(save_dir=model_dir,
         train_dir=train_path,
         )





