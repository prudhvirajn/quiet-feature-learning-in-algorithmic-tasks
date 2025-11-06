# Quiet Feature Learning in Algorithmic Tasks
This repository contains code and results accompanying the paper "Quiet Feature Learning in Algorithmic Tasks".
There are three stages to recreate the results:
1. Run scaling law experiments for tasks and input sizes. (Refer to "Running scaling laws experiments" section)
2. Train Feature Probes to detect Quiet/Loud Features. (Refer to "Training Feature Probes to detect Quiet/Loud Features across compute budget" section)
3. Causal Intervention using feature probes to determine if a feature is causally necessary to the model's performance. (Refer to "Feature Ablations" section)

## Repository Structure
```
.
├── code/               # Source code for experiments
└── data.zip            # Experimental results
```

## Installation
1. Unzip data.zip
2. Install [mamba](https://github.com/mamba-org/mamba)
3. Run the following command to install dependencies:
```
mamba create -n sage sage datasets lightning scikit-learn tokenizers torch typer wandb python=3.10
```
4. Activate environment `mamba activate sage`
5. Run `wandb init` to initialize wandb. Select the entity where you want to record logs. The project name is irrelevant so feel free to use any project name.
6. For graph tasks, install Nauty and Traces: https://pallini.di.uniroma1.it/, then generate graphs by running `geng -c 11 > graphs/graph11c.g6` or download 11-vertex simple graphs at http://users.cecs.anu.edu.au/~bdm/data/graphs.html

## Replicating experiments
Please see [README](https://github.com/prudhvirajn/quiet-feature-learning-in-algorithmic-tasks/blob/main/code/README.md) in `code` folder

## License Information
Our Transformer++ implementation is based on Meta's Llama 2 model architecture. The original Llama 2 
is licensed under the LLAMA 2 Community License (see LICENSE file). 
Our modifications include:
- Added cross-entropy loss for training
- Added support for feature ablations
- Replaced Grouped Query Attention with normal attention
- Removed KV cache optimization
- Model Arguments are determined by grid-search rather than default LLAMA model arguments

## Citing this work
If you find this work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2505.03997):
```tex
@article{naidu2025quietfeaturelearning,
  title={Quiet Feature Learning in Algorithmic Tasks},
  author={Prudhviraj Naidu and Zixian Wang and Leon Bergen and Ramamohan Paturi},
  journal={arXiv preprint arXiv:2505.03997},
  year={2025}
}
```
