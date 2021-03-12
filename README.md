# Source code for paper <A Deep Learning Account of How Language Affects Thought> https://psyarxiv.com/9xwjh
  

# Set up environment
`conda create --name myenv --file conda_envs/attn_tf22_py37.txt` 
`pip install tensorflow-gpu==2.2.0` or `conda install tensorflow-gpu==2.2.0`

 
# Replicate results
`python META_eval.py --option eval` for recreating intermediate outputs for plotting.
`python META_eval.py --option plot` for recreating result figures on the paper.


# Train models
`python META_train.py --option finegrain` for training models correspond to fine-grained labelling models
`python META_train.py --option coarsegrain` for training models correspond to coarse-grained labelling models.

