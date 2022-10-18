# GNN_Counterfactuals


## Installation

### Conda

To install the server, you need one of:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Recommended)
- [Conda](https://docs.continuum.io/anaconda/install/)


If installing for the first time you need to create the conda environment:

```bash
conda env create
```

This downloads and installs the required dependencies in an isolated environment (i.e. does not interfere with your
system's installation).

If you already have done the previous step, you might want to update the dependencies.

```bash
conda env update
```

After that you need to activate the environment:

```bash
conda activate gnn
```

If you have problems with PyTorch and pyg installation make sure you follow the following order (this worked on ubuntu22.04 LTS):
1. Install CUDA manually. For example by:
```bash
sudo apt-get -y install nvidia-cuda-toolkit
```
2. Install PyTorch (this also works with CUDA 11.5 instead of 11.3). Check version compatibility here: https://pytorch.org/get-started/previous-versions/
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
3. Install PyTorch Geometric (pyg). Check version compatibility here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```bash
conda install pyg -c pyg
```

Also install Captum for pytorch
```bash
conda install captum -c pytorch
```
Make sure you also activate the conda environment in PyCharm, a guide can be found [here](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html).

