# A Deep Learning Account of How Language Affects Thought
\[[Original paper](https://psyarxiv.com/9xwjh)\]

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
* Prepare dataset for training: 
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
