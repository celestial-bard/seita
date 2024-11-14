# ㄙㄟㄊㄚ - SeiTA
Suitable, Efficient and Intended Trajectory Attacker. Official code for *Generative Attackers Ruin Your Forecast? SeiTA: Targeted, Efficient & Realistic Adversarial Attacker for Trajectory Prediction in Autonomous Driving*.

## Usage 

### Rebuild environment

**Step 0.** Clone the repo:
```shell
& git clone https://github.com/celestial-bard/seita
& cd seita
```

**Step 1.** Create the conda environment:
```shell
& conda env create -f environment.yml
& conda activate seita
```

**Step 2.** Install [PyTorch](https://pytorch.org/) and [Lightning](https://lightning.ai/pytorch-lightning/) (You can change the url according to your CUDA version):
```shell
& pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
& pip install pytorch-lightning==2.3.3
```

**Step 3.** Install [PyG](https://pyg.org/) and its dependencies (You can change the url according to your CUDA version). This is for [QCNet](https://github.com/ZikangZhou/QCNet):
```shell
& pip install torch_geometric
& pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

**Step 4.** Install [Natten](https://shi-labs.com/natten/) (You can change the version according to your CUDA version). This is for [Forecast-MAE](https://github.com/jchengai/forecast-mae):
```shell
& pip install natten==0.17.1+torch220cu121 -f https://shi-labs.com/natten/wheels/
```

**Step 5.** Install [VideoMamba](https://github.com/OpenGVLab/VideoMamba). This is for [DeMo](https://github.com/fudan-zvg/DeMo):
```shell
& git clone https://github.com/OpenGVLab/VideoMamba.git
& cd VideoMamba
& pip install -e causal-conv1d
& pip install -e mamba
```

**Step 6.** Install Argoverse V2 api:
You can follow the guide in [Argoverse User Guide: Setup](https://argoverse.github.io/user-guide/getting_started.html#setup).
For example, use pip:
```shell
& curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
& export PATH=$HOME/.cargo/bin:$PATH
& rustup default 1.80
& pip install git+https://github.com/argoverse/av2-api#egg=av2
```

We use **Python 3.10.14 + CUDA 12.1 + PyTorch 2.2.2** to build up the environment.

### Download the data

You can follow the guide in [Argoverse User Guide: Download](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data). Either, you can directly install the [s5cmd](https://github.com/peak/s5cmd) and run the `download_av2.sh` script:
```shell
& conda install s5cmd -c conda-forge
& chmod +x ./download_av2.sh
& ./download_av2.sh motion-forecasting $HOME/datasets
```

### Preprocess the data

You'd preprocess the AV2 dataset before training. For SeiTA, it's simply to run:
```shell
& python preprocess.py -d /path/to/data/root -p -t 16
```

Also, for experiment, you'd get into `exp_methods` directory and follow their guides to preprocess data for them.

### Train the model

**Train the Reconstructor:**

We provide the Forecast-MAE as the default discriminator. Otherwise, you can surely implement other nets as disc in `exp_methods`.
```shell
& python train.py data_root=/path/to/data/root model=reconstructor
```

**Train the Generator:**
```shell
& python train.py data_root=/path/to/data/root model=generator
```

**Train the Constrainer:**
```shell
& python train.py data_root=/path/to/data/root model=constrainer
```

### Test the result
You'd attack the data first, then go to `exp_methods` and run their validate scripts.
```shell
& python write_attack.py
```

## Acknowledgement

The code is inspired by [Forecast-MAE](https://github.com/jchengai/forecast-mae), [QCNet](https://github.com/ZikangZhou/QCNet), and [AdvTrajectoryPrediction](https://github.com/zqzqz/AdvTrajectoryPrediction). Many thanks for their outstanding work.
