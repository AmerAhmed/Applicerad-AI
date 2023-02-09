import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

# import tensorflow.keras
model = tf.keras.models.load_model(os.path.abspath('model3test'))


def prediction1(model):
    """Predictions for training"""
    test_path = os.path.abspath('pictures')

    BATCH_SIZE = 10

    # Image preprocessing
    test_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1/255.
    ).flow_from_directory(
        directory=test_path,
        target_size=(20, 20),
        classes=['prediction'],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    # prediction test basche
    prediction = model.predict(test_batches)
    # Classes should convert tensors to string
    classes = ['Apple_Bad', 'Apple_Good', 'Banana_Bad', 'Banana_Good',
               'Lemon_Bad', 'Lemon_Good', 'Orange_Bad', 'Orange_Good']
    return classes[np.argmax(prediction[0])]
