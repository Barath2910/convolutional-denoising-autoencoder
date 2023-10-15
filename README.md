## CONVOLUTIONAL AUTOENCODER FOR IMAGE DENOISING...

### AIM:

To develop a convolutional autoencoder for image denoising application.

### PROBLEM STATEMENT:

Image Denoising is the process of removing noise from the Images. The noise present in the images may be caused by various intrinsic or extrinsic conditions which are practically hard to deal with. The problem of Image Denoising is a very fundamental challenge in the domain of Image processing and Computer vision. Therefore, it plays an important role in a wide variety of domains where getting the original image is really important for robust performance.

Modeling image data requires a special approach in the neural network world. The best-known neural network for modeling image data is the Convolutional Neural Network (CNN).
It can better retain the connected information between the pixels of an image. The particular design of the layers in a CNN makes it a better choice to process image data.

The CNN design can be used for image recognition/classification, or be used for image noise reduction or coloring. We can train the CNN model by taking many image samples as the inputs and labels as the outputs. We then use this trained CNN model to a new image to recognize if it is a “dog”, or “cat”, etc. CNN also can be used as an autoencoder for image noise reduction or coloring.

This program demonstrates how to implement a deep convolutional autoencoder for image denoising, mapping noisy digits images from the MNIST dataset to clean digits images.
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The training dataset in Keras has 60,000 records and the test dataset has 10,000 records. Each record has 28 x 28 pixels.

### DATASET:

![img1](https://user-images.githubusercontent.com/93427534/202606749-39e6786e-48f8-4469-b741-c1e8e3583191.png)

### CONVOLUTIONAL AUTOENCODER NEURAL NETWORK MODEL:

![img2](https://user-images.githubusercontent.com/93427534/202607417-080ebc32-7682-4ca9-a675-86ab54b3aca6.png)

![img3](https://user-images.githubusercontent.com/93427534/202608426-1a2dfc36-9810-4fd4-9ea0-f1b794aa706c.png)

### DESIGN STEPS:

#### STEP 1:

Import the necessary libraries and download the mnist dataset.

#### STEP 2:

Load the dataset and scale the values for easier computation.

#### STEP 3:

Add noise to the images randomly for the process of denoising it with the convolutional denoising autoencoders for both the training and testing sets.

#### STEP 4:

Build the Neural Model for convolutional denoising autoencoders using Convolutional, Pooling and Up Sampling layers. Make sure the input shape and output shape of the model are identical.

#### STEP 5:

Pass test data for validating manually. Compile and fit the created model.

#### STEP 6:

Plot the Original, Noisy and Reconstructed Image predictions for visualization.

#### STEP 7:

End the program.

### PROGRAM:

```python
### Developed by : Barath Kumar J
### Reg.No : 212221230012
Program to develop a convolutional autoencoder for image denoising application.

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.8
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3),activation = 'relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
                
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original:
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy:
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction:
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

```

### OUTPUT:

### ADDING NOISE TO THE MNIST DATASET:

![img4](https://user-images.githubusercontent.com/93427534/202611011-29ef99cf-57ab-4dd3-8eb7-aecb9c8c32c7.png)

### AUTOENCODER.SUMMARY():

![img5](https://user-images.githubusercontent.com/93427534/202611025-b3f24198-fe4a-4f60-84c9-37a0878029d3.png)

### ORIGINAL V/S NOISY V/S RECONSTRUCTED IMAGE:

![img6](https://user-images.githubusercontent.com/93427534/202611047-bf29669f-947a-4011-a85f-bbb5bbc36a92.png)

### RESULT:

Thus, the program to develop a convolutional autoencoder for image denoising application is developed and executted successfully.
