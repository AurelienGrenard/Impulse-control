# Neural Regression and Randomized Optimization for Impulse Control Problems

This repository contains the implementation accompanying the paper

**Neural Regression and Randomized Optimization for Impulse Control Problems**

by **Lokmane Abbas Turki, Aurélien Grenard, Idris Kharroubi and Qinghua Li**.

The code implements the neural Regression Monte Carlo methodology and randomized optimization procedures introduced in the paper for solving multidimensional impulse control problems.

## Citation

If you use this code, results, figures, or ideas from this repository in academic work, please cite the associated paper and acknowledge the original authors.

## Repository Structure

The repository contains two application examples:

```text
Harvesting/
Dividend/
```

Each directory contains:

```text
runs/
```

which stores pretrained models used in the numerical experiments presented in the paper.

In addition, each project provides four training notebooks:

```text
1D unconstrained impulse control
6D unconstrained impulse control
1D bounded impulse control
4D bounded impulse control
```

The notebooks are organized in a similar way. For a given application, all notebooks share the same code structure up to the section entitled:

```text
Learning the models
```

Only the training configuration and problem dimension differ between experiments.

## Pretrained Models

Pretrained neural networks are provided in the `runs` directories.

Users interested only in reproducing the figures and simulations may directly load the provided models and skip the training sections.

Users wishing to reproduce the complete numerical experiments may rerun the training cells from scratch.

## Environment

The experiments were conducted with:

```text
Python      3.11.2
PyTorch     2.3.1
CUDA        12.1
Matplotlib  3.10.8
NumPy       2.4.4
```

A compatible Conda environment can be created with:

```bash
conda create -n ImpulseControl python=3.11
conda activate ImpulseControl

conda install pytorch=2.3.1 pytorch-cuda=12.1 matplotlib numpy -c pytorch -c nvidia
```

## License

This code is provided for research and educational purposes.

Redistribution and modification are permitted provided that appropriate credit is given to the original authors.

## Contact

For questions regarding the implementation or the accompanying paper, please contact the authors.



```python

```
