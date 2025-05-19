## classical_chinese_poetry_sentiment_analysis
This is the source code for paper
## Abstract
Sentiment analysis of classical Chinese poetry is essential for understanding Chinese ancient literature. The majority of previous work focuses on interpreting the textual meanings to analyze the emotional expressions in poetry, often overlooking the unique prosodic and visual features inherent in poetry. In this work, we propose a multimodal framework for Classical Chinese Poetry Sentiment Analysis. Specifically, we extract sentence-level phonetic features from the poetic sentences and further incorporate regional dialect features to enrich the overall audio features. We also generate visual features for the poetry and the multimodal features are fused with the textual modality features enhanced by LLM translation through multimodal contrastive representation learning. Experimental results on two public datasets demonstrate that our method surpasses the state-of-the-art approaches.
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
