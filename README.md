# Vision Transformer Autoencoders for Unsupervised Representation Learning:  Revealing Novel Genetic Associations through Learned Sparse Attention Patterns  
This is the official repository accompanying the above paper.  

**Authors:**
**Samia Islam, Wei He, Ziqian Xie, Degui Zhi**

## Overview  
We use a vision transformer auto-encoder (ViT-AE) model to extract endophenotypes from brain imaging and conduct GWAS on UK Biobank (UKBB). 
Our approach is based on our previous work where we designed the 
[Unsupervised Deep learning-derived Imaging Phenotypes (UDIPs) model using a convolutional (CNN) autoencoder](https://www.nature.com/articles/s42003-024-06096-7). In this work, we leverage a ViT model due to a different 
inductive bias and its ability to potentially capture unique patterns through its pairwise attention mechanism. 
We derived 128 endophenotypes from average pooling and discovered 10 loci previously unreported by the CNN-based UDIP model, 
3 of which were not found in the GWAS catalog to have had any associations with brain structure. Our interpretation results 
demonstrate the ViT’s capability in capturing non-local patterns such as left- right hemisphere symmetry within brain MRI data, 
by leveraging its attention mechanism and positional embeddings. Our results highlight the advantages of transformer-based 
architectures in feature extraction and representation for genetic discovery.

**Overall pipeline**  
<img width="650" alt="image" src="https://github.com/user-attachments/assets/45350369-37ba-4356-b4df-212b532d7631" />  
Overall pipeline of the study was as follows:   
a) T1 brain MRI preprocessed by UKBB were used for training, validation and testing.   
b) ViT-AE trained by background masked mean square error (MSE) loss of non-zero patches.   
c) Genetic loci discovered by univariate GWAS and FUMA GWAS.   
d) SNP-ViT-UDIP to brain mapping performed using Perturbation-based Decoder Interpretation (PerDI).  

**ViT-AE model**  
<img width="550" alt="image" src="https://github.com/user-attachments/assets/69d09d48-7659-4613-8c5e-6b562ec9b86f" />  
The ViT-AE model generated the endophenotypes from the average pooling layer which were then used by the ViT decoder for 
prediction of output patches. With N being number of tokens and h being hidden dimension, the encoder created image embeddings 
of size N x h which were averaged across h dimensions to create the ViT-UDIPs. Empty patch embeddings with positional encoding 
were used to re-create N x h image embeddings for the decoder. Reconstruction loss is denoted by LRecon.    

**Brain image reconstruction**  
<img width="350" alt="image" src="https://github.com/user-attachments/assets/c81037f0-4332-4837-a9d5-98c55bdf6477" />  <img width="350" alt="image" src="https://github.com/user-attachments/assets/d2eb6ee6-bb8c-40f2-b645-2f160c9fbee4" />  

The figure above shows brain images of two individuals, highlighting visible differences in anatomical structure, along with their
respective reconstructed images by the ViT-AE model, where these differences appear to be preserved.   

**GWAS**  
<img width="500" alt="image" src="https://github.com/user-attachments/assets/f5da3445-0950-4042-b28d-7ae7cd596cf9" />  
Notably, we discovered 10 new loci associated with brain structure that were not previously reported by the CNN-based UDIP approach. 
Using the dbSNP tool, we investigated the lead SNPs in these loci. The previously unreported loci and the corresponding 
dimension locations of UDIPs are shown on a Manhattan plot. Chromosomes 7 and 11 each had two significantly associated loci. An example of how to 
interpret the plot is as follows: the gene SLC6A20 was identified as significantly associated with the ViT-UDIP located on dimension 94, 
indicating a relationship between the genetic expression of SLC6A20 and the feature represented in dimension 94 of the learned embeddings.

**Perturbation-based Decoder Interpretation (PerDI)**  
<img width="550" alt="image" src="https://github.com/user-attachments/assets/e7a924fc-af4b-4735-8bbb-f6cc325afa1c" />  
<img width="468" alt="image" src="https://github.com/user-attachments/assets/602381b7-0130-4024-a6ff-4e8b2114efbb" />  
A perturbation-based decoder interpretation (PerDI) step mapped the ViT-UDIPs to specific brain regions. On one of the dimensions
at the subcortical level, the thalamus region was seen highlighted on both right and left sides of the brain, along with cerebral 
white matter and cerebral cortex. At the structural level, tThe cerebellum, along with the frontal and parietal lobes, 
was highlighted, capturing a comprehensive range of structural patterns across the hindbrain and forebrain.

## Code walkthrough
[pipeline](https://github.com/ZhiGroup/UDIP-ViT/tree/main/pipeline): pipeline notebook  
[data](https://github.com/ZhiGroup/UDIP-ViT/tree/main/data): custom dataloader  
[models](https://github.com/ZhiGroup/UDIP-ViT/tree/main/models): ViT-AE models  
[training](https://github.com/ZhiGroup/UDIP-ViT/tree/main/training): model training  

## Ethics oversight
Our analysis was approved by UTHealth Houston committee for the protection of human subjects under No. HSC-SBMI-20-1323. 
UKBB has secured informed consent from the participants in the use of their data for approved research projects. 
UKBB data was accessed via approved project 24247. 

## Acknowledgements
This work was supported by grants from the National Institute on Aging U01AG070112 and R01AG081398. 

## How to Cite:
Islam, S., He, Wei, Xie, Z., & Zhi, D. (2025). Vision Transformer Autoencoders for Unsupervised Representation Learning: 
Revealing Novel Genetic Associations through Learned Sparse Attention Patterns. 
MedRxiv. [https://doi.org/10.1101/2025.03.24.25324549](https://doi.org/10.1101/2025.03.24.25324549)

## Warning
This repo is for research purpose. Using it at your own risk.
GPL-v3 license.



