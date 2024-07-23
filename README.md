## Overview
The "ARTGAN" project aims to use PyTorch and deep learning to create art that closely resembles the statistical patterns of a specific training dataset

Select a photograph or random image of a person. Using GAN, this picture serves as a blank canvas on which realistic portraits of imaginary people can be drawn. Unlike the original, these photos seem incredibly real and have no similarity to the original image. The goal of ARTGAN is to fuse creative endeavours on new data that resembles an existing dataset. ARTGAN is the process of using GAN principles to create creative output, such as pictures, paintings, or other visual media. Tasks like artistic picture synthesis, style transfer, or boosting creative characteristics in digital content development could fall under this category.

## Project Description
The dataset "Best Artworks of All Time" from Kaggle is downloaded and organized. It includes various artists and their artworks, which are then processed to be used in training the GAN. Dataset is explored to understand the distribution of artworks and artists. Visualization techniques are used to display sample images from the dataset. The images are resized, normalized, and transformed to be suitable for training the GAN. A data loader is created to handle batching and feeding images into the model. 

The GAN architecture is defined where generator creates images from random noise, and the discriminator attempts to distinguish between real and generated images. The training process involves alternating between training the discriminator and the generator. Losses for both the generator and the discriminator are calculated and used to update the model parameters. Generated images are saved at various epochs to visualize the progress of the generator. The final models for the generator and discriminator are saved for future use.

## TechStack and Libraries Requirement
- Hardware: NVIDIA GPUs cuda device and T4 GPU runtime or A good CPU and a GPU with at least 8GB memory and At Least 8GB of RAM.
- Tools: Anaconda Jupyter Notebook
- Libraries: 
➔ Python - 3.6.7
➔ Numpy - 1.16.4
➔ Tensorflow - 1.13.1
➔ Keras - 2.2.4
➔ PIL - 4.3.0
➔ cv2
➔ Matplotlib - 3.0.3
➔ torch
➔ torchvision.utils, torch.nn

## Dataset Description 
For this Project, I utilized dataset [Best Artworks of all Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) from Kaggle. This dataset consists of one csv file "artists.csv" and two folder as "images" and "resized". Image folder further consists of 50 directories of all the famous artists such as Leonardo-da-vinci, Vincent-van-gogh, Pablo picasso, etc. All these 50 directories containes all the artwork of these famous artists in JPG format. Resized folder is similar to image folder consists of all artworks sequentially but have been resized which utilizes less data for uploading and training/process would be more faster for model. 'artists.csv' column consists of name of artists, years (duration between which all the artworks were created), genre (style of paintings and sketching) and nationality (country artists belongs to).

![image](https://github.com/user-attachments/assets/96f26733-f1ef-4102-b391-4086fbb16393)
![image](https://github.com/user-attachments/assets/a7e847b5-9d34-4446-8782-712bd521159d)
![image](https://github.com/user-attachments/assets/50173d24-41eb-4854-89da-ffb6b5b02868)

## What is GANs (Generative Adversarial Networks)?
Generative Adversarial Networks (GANs) are a type of machine learning model that consists of two neural networks: a generator and a discriminator. Generator tries to creates fake data from random noise anf tries to produce data that looks like real data. Whereas, Discriminator evaluates whether the data is real or fake. and tries to distinguish between real data and data produced by the generator. The generator and the discriminator compete relentlessly in a zero-sum game to outsmart each other.

The generator and discriminator are trained together in a competitive process. generator aims to create data that can fool the discriminator and discriminator aims to correctly identify real vs. fake data. This adversarial process continues until the generator produces highly realistic data.

![image](https://github.com/user-attachments/assets/5ccce639-e0da-4f86-b79e-a3562082edfc)
<p align="center"><b>Figure 1:</b> GANs Architecture</p>

GANs have a wide range of applications: Image Generation, Data Augmentation, Image-to-Image Translation, Video Generation Super-Resolution, Text-to-Image Synthesis.

### 1.) Generator Model 
This model create synthetic images from random latent vectors and aims to produce increasingly realistic images over successive iterations. Works in tandem with the discriminator, which evolves concurrently to distinguish real from generated content. It utilizes transposed convolutional layers for upscaling latent vectors progrssively. Then, Batch normalization and ReLU activation introduce non-linearity for complex feature creation. Final Layer employs a Tanh activation function to ensure pixel values fall within the valid range of [-1, 1].

### 2.) Discriminator Model
This model is key component of the Generative Adversarial Network (GAN) for binary classification. It Distinguishes between real and generated images. Training of this model iteratively refines its ability to differentiate between real and synthetic images during adversarial training. It's architecture involves multiple layers in a nuanced process.

Input Processing processes images of size 64x64 pixels with three RGB color channels. Convolutional Layers employs layers with increasing filter depth, facilitating feature extraction and spatial downsampling. It's Operations utilizes batch normalization and leaky ReLU activation during processing. Finally, the Output Layer produces a single-channel feature map representing the probability of the input image being genuine. 

## Results
A set of evaluation criteria was developed in order to gauge the caliber of the artwork that was generated. The false image quality was good, but the fake image prediction values were not very good for 200 epochs with a learning rate of 0.0002, according to our performance calculation. The Generator (loss-g) seems to be struggling, as the loss is relatively high (4.3852). This suggests that the generated images might not resemble the real images well. The Discriminator (loss-d) is performing well, as the loss is low (0.0655). The Discriminator is effectively distinguishing between real and generated images. The Discriminator assigns a high score (0.9508) to real images, indicating that it correctly identifies them as real. Conversely, it assigns a low score (0.0121) to generated images, correctly recognizing them as fake. We therefore run it with a lower learning rate of 0.0001 and close to 80–100 epochs. The model produced artwork that was remarkably diverse and original, even though it was not always identical to human-created art.
