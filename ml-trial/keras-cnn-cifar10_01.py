from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import TensorBoard

# Download the dataset of CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Check the shapes of CIFAR-10 data to be downloaded.
print('x_train.shape: ', x_train.shape)  # (50000, 32, 32, 3)
print('x_test.shape: ', x_test.shape)    # (10000, 32, 32, 3)
print('y_train.shape: ', y_train.shape)  # (50000, 1)
print('y_test.shape: ', y_test.shape)    # (10000, 1)

# Preprocessing - exchange the scale of data
x_train = x_train/255
x_test = x_test/255

# Preprocessing - change class label to 1-hot vector
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the neural network
# Input layer + Convolutional layer 1st
model = Sequential()
model.add(
    Conv2D(
        filters=32,
        input_shape=(32, 32, 3),
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

# Convolutional layer 2nd
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

# MaxPooling layer 1st
model.add(MaxPool2D(pool_size=(2, 2)))

# Dropout layer 1st
model.add(Dropout(0.25))

# Convolutional layer 3rd
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

# Convolutional layer 4th
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)

# MaxPooling layer 2nd
model.add(MaxPool2D(pool_size=(2, 2)))

# Dropout layer 2nd
model.add(Dropout(0.25))

# Check the shape of output after Dropout layer 2nd
print(model.output_shape)

# Change the shape to 2-Demension for Dense layer
model.add(Flatten())
print(model.output_shape)

# Dense layer 1st
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(units=10, activation='softmax'))

# Learn the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb = TensorBoard(log_dir='./logs')
history_model_cifar10 = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)
