.. role:: raw-html-m2r(raw)
   :format: html



.. image:: ../logo/horizon_banner.png
   :alt: Banner


ReAgent: Applied Reinforcement Learning Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://ci.pytorch.org/jenkins/buildStatus/icon?job=horizon-master
   :target: https://ci.pytorch.org/jenkins/job/horizon-master/
   :alt: Build Status

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~

ReAgent is an open source end-to-end platform for applied reinforcement learning (RL) developed and used at Facebook.
ReAgent is built in Python and uses PyTorch for modeling and training and TorchScript for model serving. The platform contains
workflows to train popular deep RL algorithms and includes data preprocessing, feature transformation, distributed training,
counterfactual policy evaluation, and optimized serving. For more detailed information about ReAgent see the white
paper `here <https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/>`_.
The source code is available `here <https://github.com/facebookresearch/Horizon>`_.

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

Detailed instructions on how to use ReAgent can be found here: :ref:`usage`.

License
~~~~~~~~~~~~~~

ReAgent is released under a BSD license.  Find out more about it here: :ref:`license`.

.. image:: ../logo/horizon_logo.png
   :alt: Logo
   :width: 200px

Table of Contents
~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :caption: Getting Started

    Installation <installation>
    Usage <usage>

.. toctree::
    :caption: Advanced Topics

    Distributed Training <distributed>

.. toctree::
    :caption: Package Reference

    Workflow <api/ml.rl.workflow>
    Preprocessing <api/ml.rl.preprocessing>
    Simulators <api/ml.rl.simulators>
    Thrift <api/ml.rl.thrift>
    Training <api/ml.rl.training>
    All Modules <api/modules>

.. toctree::
    :caption: Others

    Github <https://github.com/facebookresearch/ReAgent>
    License <license>
