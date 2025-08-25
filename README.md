# Investigating the sensibility of interpretation methods of CLIP

This work aims to investigate the robustness of interpretability methods applied to CLIP, with an emphasis on the sensitivity of these techniques to small input perturbations, an aspect that can undermine the reliability of the generated explanations. To this end, an evaluation pipeline based on controlled perturbations was proposed, along with a set of metrics including Spearman’s rank correlation, Structural Similarity Index (SSIM), and Top-K Intersection. Nine interpretability methods were evaluated, revealing significant variability in terms of stability. Interpretation techniques such as Grad-ECLIP and CLIP Surgery showed greater robustness and semantic coherence in the face of perturbations, while approaches like RISE and Self-Attention demonstrated considerable instability. The results highlight the importance of considering not only the informativeness of the explanations, but also their robustness under different conditions.

- framework
<img width=90% src="https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/images/framework.png"/>

- visualization comparison of different XAI methods on explaining image encoder with provided text prompts. 
<img width=90% src="https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/images/examples.jpg"/>

- visualization comparison of different XAI methods on explaining both image encoder and text encoder with image-text pair. 
<img width=90% src="https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/images/examples_img_text.jpg"/>

# How to run

to-do

# Disclaimer

This project is based on the repository of [Grad-ECLIP](https://arxiv.org/abs/2502.18816) made by Chenyang Zhao.
