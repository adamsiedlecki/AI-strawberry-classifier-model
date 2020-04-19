import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

print("----------------------------")
print("Hello there! Your current configuration:")
print("----------------------------")
print ("Version of Tensorflow: "+tf.version.VERSION)
print("----------------------------")
os.system('cmd /c python -V')
print("----------------------------")
os.system('cmd /c nvcc --version')
print("----------------------------")
print("----------------------------")

train_dir = 'filtered/train'
validation_dir = 'filtered/validation'

train_truskawki_dojrzale_dir = os.path.join(train_dir, 'truskawki_dojrzale')
train_truskawki_zepsute_dir = os.path.join(train_dir, 'truskawki_zepsute')
train_brak_owocow_dir = os.path.join(train_dir, 'brak_owocow')

validation_truskawki_dojrzale_dir = os.path.join(validation_dir, 'truskawki_dojrzale')
validation_truskawki_zepsute_dir = os.path.join(validation_dir, 'truskawki_zepsute')
validation_brak_owocow_dir = os.path.join(validation_dir, 'brak_owocow')

num_truskawki_dojrzale_tr = len(os.listdir(train_truskawki_dojrzale_dir))
num_truskawki_zepsute_tr = len(os.listdir(train_truskawki_zepsute_dir))
num_brak_owocow_tr = len(os.listdir(train_brak_owocow_dir))

num_truskawki_dojrzale_val = len(os.listdir(validation_truskawki_dojrzale_dir))
num_truskawki_zepsute_val = len(os.listdir(validation_truskawki_zepsute_dir))
num_brak_owocow_val = len(os.listdir(validation_brak_owocow_dir))

total_train = num_truskawki_dojrzale_tr+num_truskawki_zepsute_tr+num_brak_owocow_tr
total_val = num_truskawki_dojrzale_val+num_truskawki_zepsute_val+num_brak_owocow_val

print("Total images: ")
print("------------------")
print("Truskawki dojrzale TRAIN: ", num_truskawki_dojrzale_tr)
print("Truskawki zepsute TRAIN: ", num_truskawki_zepsute_tr)
print("Truskawki - brak owocow TRAIN: ", num_brak_owocow_tr)
print("------------------")
print("Truskawki dojrzale VALIDATION: ", num_truskawki_dojrzale_val)
print("Truskawki zepsute VALIDATION: ", num_truskawki_zepsute_tr)
print("Truskawki - brak owocow VALIDATION: ", num_brak_owocow_val)
print("------------------")

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=45) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(sample_training_images[:5])
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
#plotImages(augmented_images)

dropout = 0.3
model = Sequential([
    Conv2D(16, 3, padding='valid', use_bias=True, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(dropout),
    Conv2D(32, 3, padding='same',use_bias=True, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same',use_bias=True, activation='relu'),
    MaxPooling2D(),
    Dropout(dropout),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('my_model.h5') 

