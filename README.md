# RISE-NER
This is the repository for Named Entity Recognition (NER) on the MultiNERD dataset. Below, you'll find the instructions on how to run this repository.

First, install the required packages:

```
pip install -r requirements.txt
```


There are three ways to run this repository. 

The first option is to run train.py and inference.py along with the given arguments in the respective files.

An example of how to fine-tune system A and run inference on the results:
```
python train.py --system A --model distilbert-base-cased --epoch 1
```
Change to ```--system B``` if system B is desired.

Then, run inference:
```
python inference.py --system distilbert-base-cased-system-A
```
Change to ```--system distilbert-base-cased-system-B``` if system B is desired.

The second option is to run all cells in the .ipynb. Remember to change the variable 'system' to A or B in the .ipynb file, depending on what you wish to use for fine-tuning and inference.

The last option is to run the .ipynb file on google colab. The necessary packages are included in the .ipynb file.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Go27SkIkwdqSvZ5Ce7RYV2dadGR87L-h?usp=sharing)
