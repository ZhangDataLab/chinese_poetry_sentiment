## chinese_poetry_sentiment
This is the source code for paper: Picturized and Recited with Dialects: A Multimodal Chinese Representation Framework for Sentiment Analysis of Classical Chinese Poetry
## Abstract
Classical Chinese poetry is a vital and enduring part of Chinese literature, conveying profound emotional resonance. Existing studies analyze sentiment based on textual meanings, overlooking the unique rhythmic and visual features inherent in poetry,especially since it is often recited and accompanied by Chinese paintings. In this work, we propose a dialect-enhanced multimodal framework for classical Chinese poetry sentiment analysis. We extract sentence-level audio features from the poetry and incorporate audio from multiple dialects,which may retain regional ancient Chinese phonetic features, enriching the phonetic representation. Additionally, we generate sentence-level visual features, and the multimodal features are fused with textual features enhanced by LLM translation through multimodal contrastive representation learning. Our framework outperforms state-of-the-art methods on two public datasets, achieving at least 2.51% improvement in accuracy and 1.63% in macro F1. We open-source the code to facilitate research in this area and provide insights for general multimodal Chinese representation。
## Prerequisites
The code has been successfully tested in the following environment.
 - Python 3.9.18
 - PyTorch 2.0.1
 - numpy 1.22.4
 - Sklearn 1.3.0
 - Pandas 2.1.3
 - Transformers 4.44.2
 - pypinyin 0.53.0
 - diffusers 0.31.0
 - opensmile 2.5.0
## Getting Started
### Prepare your data
Use data_preprocessing.ipynb to prepare data.
### Training Model
Please run following commands for training.
```python
python fspc_model.py
```
## Cite
Please cite our paper if you find this code useful for your research:
