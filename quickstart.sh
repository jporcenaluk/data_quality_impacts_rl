#!/usr/bin/env bash
set -e

{
    conda deactivate &&
    conda env remove -n citylearn -y
} || {
    echo "No existing citylearn env to remove"
}

conda env create -f environment-citylearn.yaml
conda activate citylearn
pip install -r environment-citylearn-requirements.txt
pip install -e .