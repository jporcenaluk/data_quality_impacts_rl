#!/usr/bin/env bash
set -e

cd $HOME
arch=$(uname -m)
case "$arch" in
    x86_64)
        installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        ;;
    aarch64|arm64)
        installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        ;;
    *)
        echo "Unsupported architecture: $arch"
        exit 1
        ;;
esac

install_dir="$HOME/miniconda3"
mkdir -p "$install_dir"
wget --progress=dot:giga "$installer_url" -O "$install_dir/miniconda.sh"
bash "$install_dir/miniconda.sh" -b -u -p "$install_dir"
rm "$install_dir/miniconda.sh"

#init conda
source "$install_dir/bin/activate"
conda init --all

# create env
cd -
conda env create -f environment-citylearn.yaml
conda activate citylearn
pip install -r environment-citylearn-requirements.txt
pip install -e .
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash