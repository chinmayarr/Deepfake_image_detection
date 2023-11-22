import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
# from tensorflow.keras.activations import
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# instantiating the model in the strategy scope creates the model on the TPU

Height, Width = 160, 160

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)
model = Sequential()

# Initial Convolutional Layer
model.add(Conv2D(8, (3, 3), activation=LeakyReLU(), padding="same", input_shape=(Height, Width, 3)))
model.add(BatchNormalization())

# Add more Convolutional and Pooling Layers
model.add(Conv2D(16, (3, 3), padding="same", activation=LeakyReLU()))
model.add(Conv2D(16, (3, 3), padding="same", activation=LeakyReLU()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding="same", activation=LeakyReLU()))
model.add(Conv2D(32, (3, 3), padding="same", activation=LeakyReLU()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation=LeakyReLU()))
model.add(Conv2D(64, (3, 3), padding="same", activation=LeakyReLU()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(32, activation=LeakyReLU()))
model.add(Dense(16, activation=LeakyReLU()))

# Output Layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()
train_datagen = datagen.flow_from_directory(
    '/kaggle/input/trainingset/training',
    target_size=(Height, Width),
    batch_size=64,
    class_mode='binary'
)

# Load and preprocess the validation data
validation_datagen = datagen.flow_from_directory(
    '/kaggle/input/validationset/validation',
    target_size=(Height, Width),
    batch_size=128,
    class_mode='binary'
)

test_datagen = datagen.flow_from_directory(
    '/kaggle/input/testdataset/test',
    target_size=(Height, Width),
    batch_size=64,
    class_mode='binary'
)
# model = tf.keras.models.load_model('/kaggle/input/model100/model100.h5')
history = model.fit(train_datagen, epochs=200, validation_data=validation_datagen)
model.save('model.h5')

# Plot the training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()