

<div align="center">
    <img src="title.jpg" alt="Logo" >
<h1 align="center"> Image Coloring</h1>
</div>


## 1. Problem Statement
Image coloring using Generative Adversarial Networks (GANs) is a technique where a GAN architecture is employed to add color to grayscale images. GANs consist of two neural networks, a generator and a discriminator, which are trained together in a competitive setting. The generator network takes a grayscale image as input and tries to generate a colored version of it. The goal of the generator is to produce images that are as realistic as possible. The discriminator network takes an image (either real colored image or the generated colored image) and tries to distinguish between the real and the generated images. The discriminator's goal is to correctly classify the images as real or fake.

## 3. The Proposed Method
A GAN-Based architecture is used in this task for solving the problem. 

### Overview of GAN Architecture
  
Generator: The generator network takes a grayscale image as input and tries to generate a colored version of it. The goal of the generator is to produce images that are as realistic as possible.

Discriminator: The discriminator network takes an image (either real colored image or the generated colored image) and tries to distinguish between the real and the generated images. The discriminator's goal is to correctly classify the images as real or fake.

<div align="center">
    <img src="model.jpg" alt="Logo" >
<h3 align="center"> proposed method architecture</h3>
</div>

The generator and discriminator networks are initialized with random weights. The discriminator is trained on a batch of real colored images and a batch of generated colored images (produced by the generator from grayscale inputs). The discriminator updates its weights to improve its ability to distinguish between real and fake images. The generator takes a batch of grayscale images and generates colored versions. These generated images are then fed into the discriminator. The generator updates its weights based on the discriminator's feedback, aiming to produce more realistic colored images that can fool the discriminator.


## 4. Implementation
This section delves into the practical aspects of the project's implementation.

### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for the medical image segmentation task. It includes details about the dataset source, size, composition, preprocessing, and loading applied to it.
[Dataset](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data)

### 4.2. Model
In this subsection, the architecture and specifics of the deep learning model employed for the segmentation task are presented. It describes the model's layers, components, libraries, and any modifications made to it.

### 4.3. Configurations
This part outlines the configuration settings used for training and evaluation. It includes information on hyperparameters, optimization algorithms, loss function, metric, and any other settings that are crucial to the model's performance.

### 4.4. Train
Here, you'll find instructions and code related to the training of the segmentation model. This section covers the process of training the model on the provided dataset.

### 4.5. Evaluate
In the evaluation section, the methods and metrics used to assess the model's performance are detailed. It explains how the model's segmentation results are quantified and provides insights into the model's effectiveness.

