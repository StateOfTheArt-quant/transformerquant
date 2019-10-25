# transformerquant
**transformerquant** is an open source framework for training and evaluating deep learning models in quantitative trading domain. It provides simple and extensible interfaces and abstractions for model components, contains workflows to train popular deep learning algorithms, including data preprocessing, feature transformation, distributed training, evaluation, and model serving.

**transformerquant** provides state-of-the-art general-purpose deep learning architectures to help researcher explore and expand the boundary of predictability in quantitative trading domain. More specifically, this framework is trying to transfer state-of-the-art model architectures from Computer Vision(CV), Nature Language Processing(NLP) and other domains to quantitative finance domain. Besides, we are trying to build some pretrained models to help research improve downstream tasks performance by simply fine-tuning all pre-trained parameters.

**Algorithms Supported**
1. SSA, MILA(2017), [A structured self-attentive sentence embedding](https://arxiv.org/abs/1703.03130)
2. Transformer, Google(2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. BERT, Google(2019), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

# Install

## Dependencies
 * [featurizer](https://github.com/StateOfTheArt-quant/featurizer): A define-by-run framework for data feature engineering
 
 * [pytorch](https://github.com/pytorch/pytorch): the most popular deep learning framework for model training and evaluating with strong GPU acceleration.
 
## Install from source
~~~
git clone https://github.com/StateOfTheArt-quant/transformerquant.git
cd featurizer
python setup.py install
~~~

# Quick start

# Author

Allen Yu (allen.across@gmail.com)

# License
This project following Apache 2.0 License as written in LICENSE file

Copyright 2018 Allen Yu, StateOfTheArt.quant Lab, respective Transformer contributors

