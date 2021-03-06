# GAN-Research
Graduate research project exploring image-to-image translation using CycleGAN and other generative models.

PyTorch implementation of [CycleGAN](https://arxiv.org/abs/1703.10593) trained on a subset of the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset to convert images between *smiling* and *not smiling*

Exploring how alterations to the Frechet Inception Distance (FID) evaluation metric improves the human-perceved quality of generated images.

---

### Table of Contents
1. About The Project
1.1 Built With
2. Getting Started
3. Usage
4. Roadmap
5. Contributing
6. License
7. Contact
8. Acknowledgements

---

## 1. About The Project

This project was undertaken to satisfy the ***CPSC5900 Graduate Research Project*** at [Seattle University](https://www.seattleu.edu/scieng/computer-science/)'s Computer Science department. Its main purpose is to explore field of Deep Learning and generative adversarial networks (GANS) for image generation, and their output quality metrics.

### Frechet Inception Distance (FID)

One of the most commonly used metrics is called the [Frechet Inception Distance (FID)](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) measures the distance between two distributions of high-dimentional feature maps. Theses features represent the high-level abstractions of an image which are traditionally used for image classification. For the application of GAN evaluation, FID measures the statistical distance between distributions of generated images and example images from its training dataset.

`d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))`

### Improving FID

Traditionally, these feature maps are obtained by extracting the final pooling layer activatations of [Inception V3](https://arxiv.org/abs/1512.00567) network which has been trained on 1000 categories.






---

## 2. Tools


---

## 3. Usage


---

## 4. Roadmap



--- 

## 5. Contributing


---

## 6. License


---

## 7. Contact


---

## 8. Acknowledgements

