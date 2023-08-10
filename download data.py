#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shutil


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# !wget https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_gene_expected_count.gz
get_ipython().system('wget https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz')


# In[ ]:


get_ipython().system('gzip -d tcga_gene_expected_count.gz')


# In[ ]:


get_ipython().system('ls /content/drive/MyDrive')


# In[ ]:


shutil.copy('tcga_gene_expected_count', '/content/drive/MyDrive/tcga_gene_expected_count')

