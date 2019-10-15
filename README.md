![Banner](logo/horizon_banner.png)
### Applied Reinforcement Learning @ Facebook
[![Build Status](https://ci.pytorch.org/jenkins/buildStatus/icon?job=horizon-master)](https://ci.pytorch.org/jenkins/job/horizon-master/)
---

#### Overview
ReAgent is an open source end-to-end platform for applied reinforcement learning (RL) developed and used at Facebook. ReAgent is built in Python and uses PyTorch for modeling and training and TorchScript for model serving. The platform contains workflows to train popular deep RL algorithms and includes data preprocessing, feature transformation, distributed training, counterfactual policy evaluation, and optimized serving. For more detailed information about ReAgent see the white paper [here](https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/).

The platform was once named "Horizon" but we have adopted the name "ReAgent" recently to emphasize its broader scope in decision making and reasoning.

#### Algorithms Supported
- Discrete-Action [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- Parametric-Action DQN
- [Double DQN](https://arxiv.org/abs/1509.06461), [Dueling DQN](https://arxiv.org/abs/1511.06581), [Dueling Double DQN](https://arxiv.org/abs/1710.02298)
- Distributional RL: [C51](https://arxiv.org/abs/1707.06887) and [QR-DQN](https://arxiv.org/abs/1710.10044)
- [Twin Delayed DDPG](https://arxiv.org/abs/1802.09477) (TD3)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (SAC)

#### Installation
ReAgent can be installed via. Docker or manually. Detailed instructions on how to install ReAgent can be found [here](docs/installation.rst).

#### Usage
Detailed instructions on how to use ReAgent can be found [here](docs/usage.rst).

#### License
ReAgent is released under a BSD license.  Find out more about it [here](LICENSE).

#### Citing
@article{gauci2018horizon,
  title={Horizon: Facebook's Open Source Applied Reinforcement Learning Platform},
  author={Gauci, Jason and Conti, Edoardo and Liang, Yitao and Virochsiri, Kittipat and Chen, Zhengxing and He, Yuchen and Kaden, Zachary and Narayanan, Vivek and Ye, Xiaohui},
  journal={arXiv preprint arXiv:1811.00260},
  year={2018}
}

<img src="logo/horizon_logo.png" alt="Logo" width="200"/>
