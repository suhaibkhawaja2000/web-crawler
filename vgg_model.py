from tensorflow.keras.applications.vgg16 import VGG16
import os

try:
    model = VGG16(weights=None, include_top=True)
    model.load_weights('./checkpoints/vgg16_weights.h5')
except Exception as e:
    model = VGG16(weights='imagenet', include_top=True)

    directory_path = 'checkpoints'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    model.save_weights('./checkpoints/vgg16_weights.h5')

