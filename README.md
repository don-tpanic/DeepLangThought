# A Deep Learning Account of How Language Affects Thought
\[[Original paper](https://www.tandfonline.com/doi/full/10.1080/23273798.2021.2001023)\]

### Environment setup
```
conda env create -f conda_envs/env.yml
```
 
### Reproducing results 
* This command uses pre-computed intermediate results stored in `resources_val_white/`.
* `label_type` should be set to `finegrain` or `coarsegrain`.
```
python main_eval.py -l <label_type> -f simclr -v v3.1.run12 -p True -gpu <num_gpu>
```

### Reproducing results from scratch
* Download pre-trained SimCLR (i.e., unsupervised front end) model
```
gsutil -m cp -r \
  "gs://simclr-checkpoints-tf2/simclrv2/pretrained/r50_1x_sk0/" \
  .
```
* More info about SimCLR model can be found at the [Official repo for SimCLR](https://github.com/google-research/simclr/tree/master/tf2)
  
* Prepare dataset for training (a full path to ImageNet-2012 should be set in `TRAIN/utils/data_utils.py`)
```
python data.py --model simclr
```
* Train the models
```
python main_train.py -l <label_type> -f simclr -v v3.1.run12 -r True -gpu <num_gpu>
```
* Evaluate trained models and plot results
```
python main_eval.py -l <label_type> -f simclr -v v3.1.run12 -s True -m True -p True -gpu <num_gpu>
```
