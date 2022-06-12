# gnn-wireless

Final Year Project

## Basic Steps:

- Set up a virtual env, install [IGNNITION](https://ignnition.org) via pip and other necessary packages as required.
- Run [generate_main.m](FPlinQ/generate_main.m) file to generate the dataset with MATLAB.
- Run [gen_dataset.py](gen_dataset.py) file to convert the generated mat files to serialized json (what IGNNITION expects) using NetworkX
- Set model description, training option and global variables
- Run [main.py](main.py) to start training

### Script stuff

Needs:
- dos2unix (from package repository)
- yamlpath (from python)