# Handwritten Digit Generator using DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate handwritten digit images using the MNIST dataset. The model learns patterns from real handwritten digits and generates new realistic digit images from random noise.

This project helped me understand how GANs work, including the roles of the generator and discriminator, adversarial training, and image generation using deep learning.

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab

---

## Dataset

I used the MNIST dataset, which contains handwritten digits from 0 to 9.

- Total images: 70,000
- Training images: 60,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Format: Grayscale

The model was trained using the training images to learn the distribution of handwritten digits.

---

## How the Model Works

The GAN consists of two parts:

**Generator**
- Takes a 100-dimensional random noise vector as input
- Generates a handwritten digit image

**Discriminator**
- Takes an image as input
- Predicts whether the image is real or generated (fake)

Both models are trained together so the generator improves over time and produces realistic images.

---
