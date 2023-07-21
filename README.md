# Intelligent Phenotype-detection and Gene expression profile Generation with Generative adversarial network
Gene expression analysis is valuable for cancer type classification and identifying diverse cancer phenotypes. The latest high-throughput RNA sequencing devices have enabled access to large volumes of gene expression data. However, accessing these datasets presents challenges in terms of data security and privacy. To address these issues, we propose IP3G (Intelligent Phenotype-detection and Gene expression profile Generation with Generative adversarial network), a model based on Generative Adversarial Networks. IP3G tackles two major problems: augmenting gene expression data and unsupervised phenotype discovery. By converting gene expression profiles into 2-Dimensional images and leveraging IP3G, we generate new profiles for specific phenotypes. IP3G learns disentangled representations of gene expression patterns and identifies phenotypes without annotated data.

## Implementation
Keras implementation of "Intelligent Phenotype-Detection and Gene Expression Profile Generation with Generative Adversarial Networks"

![Alt text](generator.jpg?raw=true "Generator Model")


![Alt text](discrimiantor.jpg?raw=true "Discriminator Model")

## Dataset :
TCGA  and GTEx datasets are available in this repository : https://xenabrowser.net/datapages/

Other datasets on this site are also compatible with our code.

use the "download data.ipynb" to download data.

## Gene expression to Image:
To convert gene expression profiles to images with size 128x128, use "preprocessing.ipynb"

## Training IP3G
You can use "new model.ipynb" to train IP3G model
![Alt text](samples.png?raw=true "Samole Image Generation")





