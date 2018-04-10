---
id: models
title: Supported Models
sidebar_label: Models
---

We use Deep Q Network implementations for our models. See [dqn-Atari by Deepmind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

1. [Max-Q-Learning](https://en.wikipedia.org/wiki/Q-learning) (as demonstrated in paper):
   * input: state: _s_, action _a_
   * output: scalar  _Q(s, a)_
   * update target on transition {state, action, reward, next\_state, next\_action}:
     * Q\_target(state, action) = reward + reward\_discount\_factor * max_\{possible\_next\_action} Q(next\_state, possible\_next\_action)
2. [SARSA](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action):
   * input: state _s_, action _a_
   * output: scalar _Q(s, a)_
   * update target on transition {state, action, reward, next\_state, next\_action}:
     * Q\_target(state, action) = reward + reward\_discount\_factor * Q(next\_state, next\_action)

Both of these accept discrete and parametric action inputs.

  * Discrete (but still one-hotted) action implementation: [DiscreteActionTrainer](https://github.com/facebookresearch/BlueWhale/blob/master/ml/rl/training/discrete_action_trainer.py)
  * Parametric action implementation: [ContinuousActionDQNTrainer](https://github.com/facebookresearch/BlueWhale/blob/master/ml/rl/training/continuous_action_dqn_trainer.py)
