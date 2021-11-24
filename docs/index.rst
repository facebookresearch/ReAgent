.. role:: raw-html-m2r(raw)
   :format: html



.. image:: ../logo/reagent_banner.png
   :alt: Banner


ReAgent: Applied Reinforcement Learning Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://circleci.com/gh/facebookresearch/ReAgent/tree/main.svg?style=svg
    :target: https://circleci.com/gh/facebookresearch/ReAgent/tree/main

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~

ReAgent is an open source end-to-end platform for applied reinforcement learning (RL) developed and used at Facebook.
ReAgent is built in Python and uses PyTorch for modeling and training and TorchScript for model serving. The platform contains
workflows to train popular deep RL algorithms and includes data preprocessing, feature transformation, distributed training,
counterfactual policy evaluation, and optimized serving. For more detailed information about ReAgent, please read
`releases post <https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/>`_
and `white paper <https://arxiv.org/abs/1811.00260>`_.

The source code is available here: `Source code <https://github.com/facebookresearch/ReAgent>`_.

The platform was once named "Horizon" but we have adopted the name "ReAgent" recently to emphasize its broader scope in decision making and reasoning.

Algorithms Supported
~~~~~~~~~~~~~~~~~~~~

Classic Off-Policy algorithms:

* Discrete-Action `DQN <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* Parametric-Action DQN
* `Double DQN <https://arxiv.org/abs/1509.06461>`_\ , `Dueling DQN <https://arxiv.org/abs/1511.06581>`_\ , `Dueling Double DQN <https://arxiv.org/abs/1710.02298>`_
* Distributional RL `C51 <https://arxiv.org/abs/1707.06887>`_\ , `QR-DQN <https://arxiv.org/abs/1710.10044>`_
* `Twin Delayed DDPG <https://arxiv.org/abs/1802.09477>`_ (TD3)
* `Soft Actor-Critic <https://arxiv.org/abs/1801.01290>`_ (SAC)
* `Critic Regularized Regression <https://arxiv.org/abs/2006.15134>`_ (CRR)
* `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_ (PPO)

RL for recommender systems:

* `Seq2Slate <https://arxiv.org/abs/1810.02019>`_
* `SlateQ <https://arxiv.org/abs/1905.12767>`_

Counterfactual Evaluation:

* `Doubly Robust <https://arxiv.org/abs/1612.01205>`_ (for bandits)
* `Doubly Robust <https://arxiv.org/abs/1511.03722>`_ (for sequential decisions)
* `MAGIC <https://arxiv.org/abs/1604.00923>`_

Multi-Arm and Contextual Bandits:

* `UCB1 <https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf>`_
* `MetricUCB <https://arxiv.org/abs/0809.4882>`_
* `Thompson Sampling <https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf>`_
* `LinUCB <https://arxiv.org/abs/1003.0146>`_


Others:

* `Cross-Entropy Method <http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf>`_
* `Synthetic Return for Credit Assignment <https://arxiv.org/abs/2102.12425>`_


Installation
~~~~~~~~~~~~~~~~~~~

ReAgent can be installed via. Docker or manually. Detailed instructions on how to install ReAgent can be found
here: :ref:`installation`.


Tutorial
~~~~~~~~~~~~
ReAgent is designed for large-scale, distributed recommendation/optimization tasks where we don’t have access to a simulator.
In this environment, it is typically better to train offline on batches of data, and release new policies slowly over time.
Because the policy updates slowly and in batches, we use off-policy algorithms. To test a new policy without deploying it,
we rely on counter-factual policy evaluation (CPE), a set of techniques for estimating a policy based on the actions of another policy.

We also have a set of tools to facilitate applying RL in real-world applications:


* Domain Analysis Tool, which analyzes state/action feature importance and identifies whether the problem is a suitable for applying batch RL
* Behavior Cloning, which clones from the logging policy to bootstrap the learning policy safely

Detailed instructions on how to use ReAgent can be found here: :ref:`usage`.


License
~~~~~~~~~~~~~~

| ReAgent is released under a BSD license.  Find out more about it here: :ref:`license`.
| Terms of Use - `<https://opensource.facebook.com/legal/terms>`_
| Privacy Policy - `<https://opensource.facebook.com/legal/privacy>`_
| Copyright © 2022 Meta Platforms, Inc

Citing
~~~~~~

Cite our work by:
::
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
    Usage <usage>
    RASP (Not Actively Maintained) <rasp_tutorial>

.. toctree::
    :caption: Advanced Topics

    Continuous Integration <continuous_integration>

.. toctree::
    :caption: Package Reference

    Core <api/reagent.core>
    Data <api/reagent.data>
    Gym <api/reagent.gym>
    Evaluation <api/reagent.evaluation>
    Lite <api/reagent.lite>
    MAB <api/reagent.mab>
    Model Managers <api/reagent.model_managers>
    Model Utils <api/reagent.model_utils>
    Net Builders <api/reagent.net_builder>
    Optimizers <api/reagent.optimizer>
    Models <api/reagent.models>
    Prediction <api/reagent.prediction>
    Preprocessing <api/reagent.preprocessing>
    Training <api/reagent.training>
    Workflow <api/reagent.workflow>
    All Modules <api/modules>

.. toctree::
    :caption: Others

    Github <https://github.com/facebookresearch/ReAgent>
    License <license>
