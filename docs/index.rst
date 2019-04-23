.. role:: raw-html-m2r(raw)
   :format: html



.. image:: ../logo/horizon_banner.png
   :alt: Banner


Horizon: Applied Reinforcement Learning Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://ci.pytorch.org/jenkins/buildStatus/icon?job=horizon-master
   :target: https://ci.pytorch.org/jenkins/job/horizon-master/
   :alt: Build Status

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~

Horizon is an open source end-to-end platform for applied reinforcement learning (RL) developed and used at Facebook. 
Horizon is built in Python and uses PyTorch for modeling and training and Caffe2 for model serving. The platform contains 
workflows to train popular deep RL algorithms and includes data preprocessing, feature transformation, distributed training, 
counterfactual policy evaluation, and optimized serving. For more detailed information about Horizon see the white 
paper `here <https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/>`_.
The source code is available `here <https://github.com/facebookresearch/Horizon>`_.

Algorithms Supported
~~~~~~~~~~~~~~~~~~~~


* Discrete-Action `DQN <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* Parametric-Action DQN
* `Double DQN <https://arxiv.org/abs/1509.06461>`_\ , `Dueling DQN <https://arxiv.org/abs/1511.06581>`_\ , `Dueling Double DQN <https://arxiv.org/abs/1710.02298>`_
* `DDPG <https://arxiv.org/abs/1509.02971>`_ (DDPG)
* `Soft Actor-Critic <https://arxiv.org/abs/1801.01290>`_ (SAC)

Installation
~~~~~~~~~~~~~~~~~~~

Horizon can be installed via. Docker or manually. Detailed instructions on how to install Horizon can be found
here: :ref:`installation`.

Usage
~~~~~~~~~~~~

Detailed instructions on how to use Horizon can be found here: :ref:`usage`.

License
~~~~~~~~~~~~~~

Horizon is released under a BSD license.  Find out more about it here: :ref:`license`.

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
    Training <api/ml.rl.training>
    Github <https://github.com/facebookresearch/Horizon>
    License <license>
