from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Download VGG16 (Pre-trained model).
model = VGG16()

# Check the summary of VGG16 to be downloaded.
model.summary()

# Load and re-size the images for network input
img_dog = load_img('img/dog.jpg', target_size=(224, 224))
img_cat = load_img('img/cat.jpg', target_size=(224, 224))

# Exchange dog/cat image from Pillow data format to numpy.ndarray.
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)

# Centering img color channel and change the order of them
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)

# Merge the img for network input as array.
arr_input = np.stack([arr_dog, arr_cat])

# Check the shape of input data.
print('Shape of arr_input:', arr_input.shape)

# Prediction of input images.
probs = model.predict(arr_input)
print('Shape of probs:', probs.shape)
print(probs)

# Decode prediction to class name and pick up 1-5 classes by high percentage order.
results = decode_predictions(probs)
print(results[0])
print(results[1])

