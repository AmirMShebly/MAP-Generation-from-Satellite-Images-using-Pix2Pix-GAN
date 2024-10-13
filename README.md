# Map Generation from Satellite Images using Pix2Pix GAN


## Project Overview
This project focuses on generating high-quality maps from satellite images using the Pix2Pix Generative Adversarial Network (GAN). Pix2Pix is a popular conditional GAN framework designed for image-to-image translation tasks. The goal is to automatically generate detailed and accurate maps from satellite imagery, which can be beneficial for urban planning, disaster management, and geographic analysis.

## Task Description
Given an input satellite image, the trained Pix2Pix GAN model generates a corresponding map image. The model learns a mapping between satellite images and their map counterparts through supervised learning on paired image data. Each satellite image is paired with its corresponding ground truth map during training, allowing the model to capture the structural and visual differences between satellite imagery and map representations.

## Dataset
The dataset  used for training and evaluation is a part of pix2pix dataset consists of pairs of satellite images and their corresponding maps. These images are organized into seven folders, with each folder containing paired images.

## Model Architecture
original paper:
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
The Pix2Pix model consists of two key components:
Generator: A U-Net-based architecture that translates input satellite images into maps.
Discriminator: A PatchGAN-based discriminator that classifies whether the generated map looks realistic when compared to the ground truth map.

## Results
Here are some example results:


Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 ![satellite_image_1](https://github.com/user-attachments/assets/e9cd3118-d446-4dfd-a662-ce65a8ca556a) | ![generated_map_1](https://github.com/user-attachments/assets/366f7cd0-afea-475a-8b1b-ef71c83ab211)
 




Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 | 




Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 | 





Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 | 





Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 | 




Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 | 

