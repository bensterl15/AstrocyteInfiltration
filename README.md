# README.md
## Environment installation

```bash
conda create -n NewEnviro python=3.8
conda activate NewEnviro
pip install gunpowder
conda install pytorch
git clone https://github.com/funkelab/funlib.learn.torch
cd funlib.learn.torch
pip install -r requirements.txt
pip install .
conda install -c conda-forge zarr

```
