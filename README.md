# Investigating the sensibility of interpretation methods of CLIP

This is a ongoing work with the objective of find how sensitive the interpretation methods of VLMs are, especially CLIP as it is the most influential one. Future updates will be posted here.

Grad-Eclip is a straightforward and easy-to-implement method to generate visual explanation heat maps for transformer-based CLIP. It can be applied on both image and text branch. The framework and results are shown here:

- framework
<img width=90% src="https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/images/framework.png"/>

- visualization comparison of different XAI methods on explaining image encoder with provided text prompts. 
<img width=90% src="https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/images/examples.jpg"/>

- visualization comparison of different XAI methods on explaining both image encoder and text encoder with image-text pair. 
<img width=90% src="https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/images/examples_img_text.jpg"/>

# Disclaimer

This project is based on the repository of [Grad-ECLIP](https://arxiv.org/abs/2502.18816) made by Chenyang Zhao. Please take a look at his brilliant work.
