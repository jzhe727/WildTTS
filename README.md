# WildTTS
Course project for ENSF 619 

### Directory structure

# Installation 
## Backend

Prequisites: 
Conda
Git (with LFS): [tutorial for installation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

* Clone the repository.

```
git clone --recursive git@github.com:jzhe727/WildTTS.git
cd WildTTS
git submodule update --init --recursive
```
* Create a new Conda environment named wildtts and activate it
```
conda create -n wildtts -y python=3.10
conda activate wildtts
```

* Install ffmpeg version 6, 5, or 4 for torchaudio if it is not currently installed.
Warning: do not do this in the base environment.

```
conda install -c conda-forge 'ffmpeg<7'
```
* Install requirements for Cosyvoice 2. Can also follow the README in the CosyVoice repository.

```
cd backend/Cosyvoice
pip install -r requirements.txt 
```

* Obtain pretrained model checkpoint. Here we use Git LFS to download CosyVoice 2 checkpoint.
```
mkdir -p pretrained_models

git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git \
pretrained_models/CosyVoice2-0.5B
```

* Test run

Running template.py (in the backend folder) *should* generate some reasonable audio.
