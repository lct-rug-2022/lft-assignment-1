# Learning From Data Assignment 1: Multi-domain Text Reviews Category Prediction


### Installation 

```shell
python -m venv .venv
sourse .venv/bin/activate
pip isntall -U -r requitements.txt
```

### Training

We provide `train.py` script accepting train and test dataset and return model score and save pipeline as `--model_file`
```shell
python train.py --help
```

### Testing 

We provide `test.py` script and `pipeline.pkl`with saved model to run it on any test data
```shell
python test.py --help
```


### Data

Dataset is provided in 2 variants: `datasets/reviews.txt` and already split `datasets/test.txt`/`datasets/train.txt`. Data already tokenized.


### Experiments 

Experiments are available as ipython notebooks at `experiments` folder 
