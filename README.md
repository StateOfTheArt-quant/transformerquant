# transformerquant
**transformerquant** is an open source framework for training and evaluating deep learning models in quantitative trading domain.  It contains workflows to train popular deep learning algorithms, including data preprocessing, feature transformation, distributed training, evaluation, and model serving.

**transformerquant** provides state-of-the-art general-purpose deep learning architectures to help researcher explore and expand the boundary of predictability in quantitative trading domain. More specifically, this framework is trying to transfer state-of-the-art model architectures from Computer Vision(CV), Nature Language Processing(NLP) and other domains to quantitative finance domain. Besides, we are trying to build some pretrained models to help research improve downstream tasks performance by simply fine-tuning all pre-trained parameters.

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



