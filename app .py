#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="DCGAN Handwritten Digit Generator", page_icon="✏", layout="centered")

# Title & description
st.title("🖋️ DCGAN - Handwritten Digit Generator")
st.write("Generate handwritten digits using a Deep Convolutional GAN trained on the MNIST dataset.")

# Load model
@st.cache_resource
def load_generator():
    return load_model("generator_model.keras", compile=False)

generator = load_generator()

# Generate images function
def generate_images(num_images=16):
    noise = np.random.normal(0, 1, (num_images, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    return gen_imgs

# Button to generate
if st.button("Generate Digits"):
    imgs = generate_images()
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    st.pyplot(fig)

# Footer
st.markdown("**Dataset:** MNIST | **Model:** DCGAN | **Made by:** Your Name")


# In[ ]:




