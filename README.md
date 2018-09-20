# Horizon
### Applied Reinforcement Learning @ Facebook
---

#### Installation

##### Linux (Ubuntu)

Clone repo:
```
git clone https://github.com/facebookresearch/BlueWhale.git
cd BlueWhale/
```
Run install script:
```
bash linux_install.sh
```

Install appropriate PyTorch 1.0 nightly build into the virtual environment:
```
. env/bin/activate

# For CPU build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# For CUDA 9.0 build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# For CUDA 9.2 build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
```

#### Usage

Set PYTHONPATH to access caffe2 and import our modules:
```
export PYTHONPATH=/usr/local:./:$PYTHONPATH
```

##### Online RL Training
Horizon supports online training environments for model testing. To run train a model on OpenAI Gym, simply run:
```
python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_v0.json
```

##### Batch RL Training
Horizon also supports training where datasets are already generated.

TBD (fill out long example section)
