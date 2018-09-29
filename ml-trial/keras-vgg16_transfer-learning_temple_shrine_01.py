from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
from datetime import datetime
import json
import pickle
import math
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from utils import load_random_imgs, show_test_samples

# Convert VGG16 model to Sequential model.
# trainable -> false in first 15 layers.
# Add Flatten, Dense for learning, Dropout, and Dense for output.
def build_transfer_model(vgg16):
    model = Sequential(vgg16.layers)

    for layer in model.layers[:15]:
        layer.trainable = False

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

# Download and create VGG16 model without Output layer.
vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))

# Check the summary VGG16 model without Output layer.
vgg16.summary()

# Create model as Sequential model from VGG16 by `build_transfer_model` method.
model = build_transfer_model(vgg16)

# Compile the model.
model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
)

# Check the model summary after adding new layers.
model.summary()

# Create image generator.
idg_train = ImageDataGenerator(
    rescale=1/255.,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

# Create iterator for training from image generator.
img_itr_train = idg_train.flow_from_directory(
    'img/shrine_temple/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Create iterator for validation from image generator.
img_itr_validation = idg_train.flow_from_directory(
    'img/shrine_temple/validation',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Make directory for saving model, class labels, loss, and weights.
model_dir = os.path.join(
    'models',
    datetime.now().strftime('%y%m%d_%H%M')
)
os.makedirs(model_dir, exist_ok=True)
print('model_dir:', model_dir)
dir_weights = os.path.join(model_dir, 'weights')
os.makedirs(dir_weights, exist_ok=True)

# Save the model to model.json.
model_json = os.path.join(model_dir, 'model.json')
with open(model_json, 'w') as f:
    json.dump(model.to_json(), f)

# Save the class label info to classes.pkl.
model_classes = os.path.join(model_dir, 'classes.pkl')
with open(model_classes, 'wb') as f:
    pickle.dump(img_itr_train.class_indices, f)

# Define and calculate each value for learning.
batch_size = 16
step_per_epoch = math.ceil(
    img_itr_train.samples/batch_size
)
validation_steps = math.ceil(
    img_itr_validation.samples/batch_size
)

# Create callback of the model of weights.
cp_filepath = os.path.join(dir_weights, 'ep_{epoch:02d}_ls_{loss:.1f}.h5')
cp = ModelCheckpoint(
    cp_filepath,
    monitor='loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='auto',
    period=5
)

# Create callback of the value of loss.
csv_filepath = os.path.join(model_dir, 'loss.csv')
csv = CSVLogger(csv_filepath, append=True)

# Learn the model by fit_generator() method.
n_epoch = 1
history = model.fit_generator(
    img_itr_train,
    steps_per_epoch=step_per_epoch,
    epochs=n_epoch,
    validation_data=img_itr_validation,
    validation_steps=validation_steps,
    callbacks=[cp, csv]
)

# Predict the test data using the model to be learned.
test_data_dir = 'img/shrine_temple/test/unknown'
x_test, true_labels = load_random_imgs(
    test_data_dir,
    seed=1
)
x_test_preproc = preprocess_input(x_test.copy())/255.
probs = model.predict(x_test_preproc)
print(probs)

# Display the test sample data.
print(show_test_samples(
    x_test, probs,
    img_itr_train.class_indices,
    true_labels
))
