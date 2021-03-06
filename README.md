# GAN-Research
Graduate research project exploring image-to-image translation using CycleGAN and other generative models.

PyTorch implementation of [CycleGAN](https://arxiv.org/abs/1703.10593) trained on a subset of the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset to convert images between *smiling* and *not smiling*

Exploring how alterations to the Frechet Inception Distance (FID) evaluation metric improves the human-perceved quality of generated images.

---

### Table of Contents
1. About The Project
3. Tools
4. Usage
5. Roadmap
6. Contributing
7. License
8. Contact
9. Acknowledgements

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

- Python3

- PyTorch for GAN model development. 

- [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) for access to GPU-accelerated interactive notebooks to train models and execute scripts.


---

## 3. Usage

Use Colab notebooks to experiment with GAN model alterations and train model weights using Colab's GPU acceleration. Then transfer stable code into appropriate modules to reuse for future experiments. Keep Colab notebooks focused on single experiment by preparing the environment, loading the data, and performing the calculations. Keep data exploration and formal experimentation files separate to avoid confusion.




---

## 4. Roadmap





---

## 6. License

MIT License

Copyright (c) 2021 Eric Nunn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.





---

## 7. Contact

Eric Nunn

[LinkedIn](https://www.linkedin.com/in/eric-j-nunn/)

[Twitter](https://twitter.com/EricNunn11)




---

## 8. Acknowledgements

Faculty Advisor - Dr. Shadrokh Samavi

[Google Scholar](https://scholar.google.com/citations?user=Hj3vz2YAAAAJ&hl=en)






