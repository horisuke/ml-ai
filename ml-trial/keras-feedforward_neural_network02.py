from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense

# Download the dataset of MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing - exchange the scale of data
x_train = x_train.reshape(60000, 784)
x_train = x_train/255
x_test = x_test.reshape(10000, 784)
x_test = x_test/255

# Preprocessing - change class label to 1-hot vector
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the neural network
input = Input(shape=(784, ))
middle = Dense(units=64, activation='relu')(input)
output = Dense(units=10, activation='softmax')(middle)
model = Model(inputs=input, outputs=output)

# Learn the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb = TensorBoard(log_dir='./logs')
history_adam = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)
