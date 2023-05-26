import cv2 as cv
import numpy as nm
import matplotlib.pyplot as plt
import numpy as np
from keras import datasets, layers, models
from keras.datasets import cifar10

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'ship', 'Truck']

fig, axs = plt.subplots(4, 4)
for i in range(16):
    axs[i//4, i%4].imshow(training_images[i], cmap=plt.cm.binary)
    axs[i//4, i%4].set_xticks([])
    axs[i//4, i%4].set_yticks([])
    axs[i//4, i%4].set_xlabel(class_names[training_labels[i, 0]])

plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')
