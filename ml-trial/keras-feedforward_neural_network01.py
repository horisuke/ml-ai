from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard

# Download the dataset of Boston house-prices
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Check the shapes of MNIST data to be downloaded.
print('x_train.shape: ', x_train.shape)  # (60000, 28, 28)
print('x_test.shape: ', x_test.shape)    # (10000, 28, 28)
print('y_train.shape: ', y_train.shape)  # (60000, )
print('y_test.shape: ', y_test.shape)    # (10000, )

# Preprocessing - exchange the scale of data
x_train = x_train.reshape(60000, 784)
x_train = x_train/255
x_test = x_test.reshape(10000, 784)
x_test = x_test/255

# Preprocessing - change class label to 1-hot vector
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the neural network
# Input layer -> Hidden layer
model = Sequential()
model.add(
    Dense(
        units=64,
        input_shape=(784, ),
        activation='relu'
    )
)

# Hidden layer -> Output layer
model.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

# Learn the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb = TensorBoard(log_dir='./logs')
history_adam=model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)