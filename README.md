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
 ![satellite_image_2](https://github.com/user-attachments/assets/41eae2cb-dea0-4ead-94c3-161b37b16657) | ![generated_map_2](https://github.com/user-attachments/assets/dad572fd-f186-4e97-a793-c9180e79f295)





Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 ![satellite_image_3](https://github.com/user-attachments/assets/a7d561cb-aafa-49a8-ba53-5f2749b79234) | ![generated_map_3](https://github.com/user-attachments/assets/41ee4971-efe6-422f-a1ae-1743f36b4719)
 





Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 ![satellite_image_4](https://github.com/user-attachments/assets/3de53542-7632-48ef-aecb-9f1a1dcfb43f) | ![generated_map_4](https://github.com/user-attachments/assets/97292c0a-f07d-43a0-8dbd-774f49bdb0f2)






Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 ![satellite_image_5](https://github.com/user-attachments/assets/8f0a6543-bd9f-4a9e-b106-a645bbdfab70) | ![generated_map_5](https://github.com/user-attachments/assets/0768f581-9961-4847-a136-19484b658c95)
 




Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 ![satellite_image_6](https://github.com/user-attachments/assets/0e04d0fb-47cf-4540-9639-a819be8becd7) | ![generated_map_6](https://github.com/user-attachments/assets/2a7aef97-810f-4e3a-90c9-fda5e85cf34f)
 




Satellite Image                     |  Generated Map
:----------------------------------:|:-------------------------:
 ![satellite_image_7](https://github.com/user-attachments/assets/e2e4b192-cce4-4783-93e0-592d5b8cee94) | ![generated_map_7](https://github.com/user-attachments/assets/9dde1f0a-9a3e-43c9-bca7-1f95a3265c45)


