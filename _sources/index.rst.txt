.. role:: raw-html-m2r(raw)
   :format: html



.. image:: ../logo/reagent_banner.png
   :alt: Banner


ReAgent: Applied Reinforcement Learning Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://circleci.com/gh/facebookresearch/ReAgent/tree/master.svg?style=svg
    :target: https://circleci.com/gh/facebookresearch/ReAgent/tree/master

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~

ReAgent is an open source end-to-end platform for applied reinforcement learning (RL) developed and used at Facebook.
ReAgent is built in Python and uses PyTorch for modeling and training and TorchScript for model serving. The platform contains
workflows to train popular deep RL algorithms and includes data preprocessing, feature transformation, distributed training,
counterfactual policy evaluation, and optimized serving. For more detailed information about ReAgent see the white
paper here: `Platform <https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/>`_.

The source code is available here: `Source code <https://github.com/facebookresearch/ReAgent>`_.

The platform was once named "Horizon" but we have adopted the name "ReAgent" recently to emphasize its broader scope in decision making and reasoning.

Algorithms Supported
~~~~~~~~~~~~~~~~~~~~


* Discrete-Action `DQN <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* Parametric-Action DQN
* `Double DQN <https://arxiv.org/abs/1509.06461>`_\ , `Dueling DQN <https://arxiv.org/abs/1511.06581>`_\ , `Dueling Double DQN <https://arxiv.org/abs/1710.02298>`_
* Distributional RL `C51 <https://arxiv.org/abs/1707.06887>`_\ , `QR-DQN <https://arxiv.org/abs/1710.10044>`_
* `Twin Delayed DDPG <https://arxiv.org/abs/1802.09477>`_ (TD3)
* `Soft Actor-Critic <https://arxiv.org/abs/1801.01290>`_ (SAC)

Installation
~~~~~~~~~~~~~~~~~~~

ReAgent can be installed via. Docker or manually. Detailed instructions on how to install ReAgent can be found
here: :ref:`installation`.

Usage
~~~~~~~~~~~~

The ReAgent Serving Platform (RASP) tutorial covers serving and training models and is available here: :ref:`rasp_tutorial`.

Detailed instructions on how to use ReAgent can be found here: :ref:`usage`.

License
~~~~~~~~~~~~~~

ReAgent is released under a BSD license.  Find out more about it here: :ref:`license`.

Citing
~~~~~~

@article{gauci2018horizon,
  title={Horizon: Facebook's Open Source Applied Reinforcement Learning Platform},
  author={Gauci, Jason and Conti, Edoardo and Liang, Yitao and Virochsiri, Kittipat and Chen, Zhengxing and He, Yuchen and Kaden, Zachary and Narayanan, Vivek and Ye, Xiaohui},
  journal={arXiv preprint arXiv:1811.00260},
  year={2018}
}

Table of Contents
~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :caption: Getting Started

    Installation <installation>
    Tutorial <rasp_tutorial>
    Usage <usage>

.. toctree::
    :caption: Advanced Topics

    Distributed Training <distributed>
    Continuous Integration <continuous_integration>

.. toctree::
    :caption: Package Reference

    Evaluation <api/ml.rl.evaluation>
    Models <api/ml.rl.models>
    Prediction <api/ml.rl.prediction>
    Preprocessing <api/ml.rl.preprocessing>
    Readers <api/ml.rl.readers>
    Simulators <api/ml.rl.simulators>
    Training <api/ml.rl.training>
    Workflow <api/ml.rl.workflow>
    All Modules <api/modules>

.. toctree::
    :caption: Others

    Github <https://github.com/facebookresearch/ReAgent>
    License <license>
