## Overview

## Project Description


## What is done in this Project?

## TechStack and Libraries Requirement
● NVIDIA GPUs cuda device and T4 GPU runtime.
● A good CPU and a GPU with at least 8GB memory.
● At Least 8GB of RAM.
● Anaconda Jupyter Notebook
● Required libraries for Python along with their version
numbers used while making & testing of this project
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
![image](https://github.com/user-attachments/assets/5ccce639-e0da-4f86-b79e-a3562082edfc)
<p align="center"><b>Figure 1:</b> GANs Architecture</p>

### 1.) Generator Model 
This model create synthetic images from random latent vectors and aims to produce increasingly realistic images over successive iterations. Works in tandem with the discriminator, which evolves concurrently to distinguish real from generated content. It utilizes transposed convolutional layers for upscaling latent vectors progrssively. Then, Batch normalization and ReLU activation introduce non-linearity for complex feature creation. Final Layer employs a Tanh activation function to ensure pixel values fall within the valid range of [-1, 1].

### 2.) Discriminator Model
This model is key component of the Generative Adversarial Network (GAN) for binary classification. It Distinguishes between real and generated images. Training of this model iteratively refines its ability to differentiate between real and synthetic images during adversarial training. It's architecture involves multiple layers in a nuanced process.

Input Processing processes images of size 64x64 pixels with three RGB color channels. Convolutional Layers employs layers with increasing filter depth, facilitating feature extraction and spatial downsampling. It's Operations utilizes batch normalization and leaky ReLU activation during processing. Finally, the Output Layer produces a single-channel feature map representing the probability of the input image being genuine. 

## Results
A set of evaluation criteria was developed in order to gauge the caliber of the artwork that was generated. The false image quality was good, but the fake image prediction values were not very good for 200 epochs with a learning rate of 0.0002, according to our performance calculation. The Generator (loss-g) seems to be struggling, as the loss is relatively high (4.3852). This suggests that the generated images might not resemble the real images well. The Discriminator (loss-d) is performing well, as the loss is low (0.0655). The Discriminator is effectively distinguishing between real and generated images. The Discriminator assigns a high score (0.9508) to real images, indicating that it correctly identifies them as real. Conversely, it assigns a low score (0.0121) to generated images, correctly recognizing them as fake. We therefore run it with a lower learning rate of 0.0001 and close to 80–100 epochs. The model produced artwork that was remarkably diverse and original, even though it was not always identical to human-created art.
