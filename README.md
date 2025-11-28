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
cd backend/CosyVoice
pip install -r requirements.txt 
```

* Obtain pretrained model checkpoint. Here we use Git LFS to download CosyVoice 2 checkpoint.
```
mkdir -p pretrained_models

git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git \
pretrained_models/CosyVoice2-0.5B
```

Additionally install CUDA runtime for GPU inference
```
conda install cuda -c nvidia/label/cuda-12.1.0 -y
```

For fish-audio-sdk

```
pip install fish-audio-sdk

```
* Test run needs an API key

Running cosyvoice_test.py (in the backend folder) *should* generate some reasonable audio.

## Evaluation

To keep dependencies isolated, we will make multiple environments and change between them for generation vs. evaluation.

Environment names must be exact for the script to work.

```
conda create -n wildtts-versa python=3.10 -y
conda activate wildtts-versa
```

```
cd eval/versa

pip install .
```

Install CUDA runtime for GPU metric calculation
```
conda install cuda -c nvidia/label/cuda-12.8.0 -y
```

### Testing the metrics calculater
```
conda activate wildtts-versa

cd eval

PRED_DIR=metrics_testing ./eval_versa.sh comprehensive --verbose"
```