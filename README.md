# Caffe2 Reinforcement Learning Models

How would you teach a robot to balance a pole? Or safely land a space ship? Or even to walk?

Using reinforcement learning (RL), you wouldn't have to teach it how to do any of these things: only what to do. RL formalizes our intuitions about trial and error – agents take actions, experience feedback, and adjust their behavior accordingly.

An agent may start with awful performance: the cart drops the pole immediately; when the space ship careens left, it tilts further; the walker can't take one step without falling. But with experience from exploration and failure, it learns. Soon enough, the agent is behaving in a way you never explicitly told it to, and is achieving the goals you implicitly set forth. It takes actions that optimize for the reward system you designed, often coming up with solutions and employing strategies you hadn't thought of.

While historically, RL has been primarily used in the context of robotics and game-playing, it can be employed in a variety of problem spaces. At Facebook, we're working on using RL at scale: suggesting people you may know, notifying you about page updates, personalizing our video bitrate serving, and more.

Advances in RL theory, including the advent of Deep Query Networks and Deep Actor-Critic models, allow us to use function approximation to approach problems with large state and action spaces.  This project, called RL_Caffe2, contains Deep RL implementations built on [caffe2](http://caffe2.ai/). We provide support for running them inside [OpenAI Gym](gym.openai.com).

# Requirements

### Recommended: Anaconda

For mac users, we recommend using [Anaconda](https://www.continuum.io/downloads) instead of the system implementation of python. Install anaconda and verify you are using anaconda's version of python before installing other dependencies: `which python` should yield an anaconda path.

### Caffe2

RL_Caffe2 runs on any platform that supports caffe2. To install caffe2, follow this tutorial: [Installing Caffe2](https://caffe2.ai/docs/getting-started.html).


### Thrift
[Thrift](https://github.com/facebook/fbthrift) is Facebook's RPC framework.
```
brew install thrift
```

### OpenAI Gym

Running models in OpenAI Gym environments requires platforms with OpenAI Gym support. Windows support for OpenAI Gym is being tracked [here](https://github.com/openai/gym/issues/11).

OpenAI Gym can be installed using [pip](https://pypi.python.org/pypi/pip) which should come with your python installation in the case of linux or with anaconda in the case of OS/X.  To install the basic environments ([classic control](https://gym.openai.com/envs#classic_control), [toy text](https://gym.openai.com/envs#toy_text), and [algorithmic](https://gym.openai.com/envs#algorithmic)), run:
```
pip install gym
```

To install [all environments](https://gym.openai.com/envs/), run this instead:
```
pip install "gym[all]"
```

# Installation and setup

Clone from source:
```
git clone https://github.com/caffe2/reinforcement-learning-models
```

To make thrift accessible to our system, run from within the root directory:
```
thrift --gen py --out . ml/rl/thrift/core.thrift
```

To access caffe2 (if you installed from) and import our modules successfuly:
```
export PYTHONPATH=/usr/local:$PYTHONPATH
```

# Running Unit Tests

From within the root directory, run all of our unit tests with:
```
python -m unittest discover
```

To run a specific unit test:
```
python -m unittest <path/to/unit_test.py>
```

# Running Models in OpenAI Gym

You can run RL models of your specification on OpenAI Gym environments of your choice. Right now, we only support environments that supply `Box(x, )` state representations and require `Discrete(y)` action inputs.

### Quickstart

```
python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/maxq_cartpole_v0.json
```

The [run_gym.py](https://github.com/caffe2/reinforcement-learning-models/tree/master/ml/rl/test/gym/run_gym.py) script will construct an RL model and run it in an OpenAI Gym environemnt, periodically reporting scores averaged over several trials. In general, you can run RL models in OpenAI Gym environments with:

```
python ml/rl/test/gym/run_gym.py -p <parameters_file> [-s <score_bar>] [-g <gpu_id>]
```
* **parameters_file**: Path to your JSON parameters file
* **score_bar** (optional): Scalar score you hope to achieve. Once your model scores at least *score_bar* well averaged over 100 test trials, training will stop and the script will exit. If left empty, training will continue until you compplete `num_iterations` iterations (see details on parameters in the next section).
* **gpu_id** (optional): If set to your machine's GPU id (typically `0`), the model will run its training and inference on your GPU. Otherwise it will use your CPU.

Feel free to create your own parameter files to select different environments and change model parameters. The success criteria for different environments can be found [here](https://gym.openai.com/envs). We currently supply default arguments for the following environments:

* [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) environment: [maxq\_cartpole\_v0.json](https://github.com/caffe2/reinforcement-learning-models/tree/master/ml/rl/test/gym/maxq_cartpole_v0.json)
* [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment: [maxq\_cartpole\_v1.json](https://github.com/caffe2/reinforcement-learning-models/tree/master/ml/rl/test/gym/maxq_cartpole_v1.json)
* [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) environment: [maxq\_lunarlander\_v2.json](https://github.com/caffe2/reinforcement-learning-models/tree/master/ml/rl/test/gym/maxq_lunarlander_v2.json)


### Modifying the parameters file

As an example, The Cartpole-v0 default parameter file we supply specifies the use of an RL model whose backing neural net has 5 layers:

```json
{
    "env": "CartPole-v0",
    "rl": {
        "reward_discount_factor": 0.99,
        "target_update_rate": 0.1,
        "reward_burnin": 10,
        "maxq_learning": 1,
        "epsilon": 0.2
    },
    "training": {
        "layers": [-1, 256, 128, 64, -1],
        "activations": ["relu", "relu", "relu", "linear"],
        "minibatch_size": 128,
        "learning_rate": 0.005,
        "optimizer": "ADAM",
        "learning_rate_decay": 0.999
    },
    "run_details": {
        "num_episodes": 301,
        "train_every": 10,
        "train_after": 10,
        "test_every": 100,
        "test_after": 10,
        "num_train_batches": 100,
        "train_batch_size": 1024,
        "avg_over_num_episodes": 100,
        "render": 0,
        "render_every": 100
    }
}
```

You can supply a different JSON parameter file, modifying the fields to your liking.

* **env**: The OpenAI gym environment to use
* **rl**
  * **reward\_discount\_factor**: A measure of how quickly the model's target network updates
  * **target\_update\_rate**: A measure of how quickly the model's target network updates
  * **reward_burnin**: The iteration after which to use the model's target network to construct target values
  * **maxq_learning**: 1 for Q-learning, 0 for SARSA
  * **epsilon**: Fraction of the time the agent should select a random action during training
* **training**
  * **layers**: An array whose ith entry specifies the number of nodes in the ith layer of the Neural Net. Use `-1` for the input and output layers; our models will fill in the appropriate values based on your choice of environment
  * **activations**: A array whose ith entry specifies the activation function to use between the ith and i+1th layers. Valid choices are `"linear"` and `"relu"`. Note that this array should have one fewer entry than your entry for *layers*
  * **minibatch_size**: The number of transitions to train the Neural Net on at a time. This will not effect the total number of datapoints supplied. In general, lower/higher minibatch sizes perform better with lower/higher learning rates
  * **learning_rate**: Learning rate for the neural net
  * **optimizer**: Neural net weight update algorithm. Valid choices are `"SGD"`, `"ADAM"`, `"ADAGRAD"`, and `"FTRL"`
  * **learning\_rate\_decay**: Factor by which the learning rate decreases after each training minibatch
* **run_details**
  * **num_episodes**: Number of episodes run the mode and to collect new data over
  * **train_every**: Number of episodes between each training cycle
  * **train_after** Number of episodes after which to enable training
  * **test_every**: Number of episodes between each test cycle
  * **test_after**: Number of episodes after which to enable testing
  * **num\_train\_batches**: Number of batches to train over each training cycle
  * **train\_batch\_size**: Number of transitions to include in each training batch. Note that these will each be further broken down into minibatches of size *minibatch_size*
  * **avg\_over\_num\_episodes**: Number of episodes to run every test cycle. After each cycle, the script will report an average over the scores of the episodes run within it.The typical choice is `100`, but this should be set according to the [success criteria](https://gym.openai.com/envs) for your environment
  * **render**: Whether or not to render the OpenAI environment in training and testing episodes. Note that some environments don't support rendering
  * **render_every**: Number of episodes between each rendered episode



# Supported Models:

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

  * Discrete (but still one-hotted) action implementation: [DiscreteActionTrainer](https://github.com/caffe2/reinforcement-learning-models/blob/master/ml/rl/training/discrete_action_trainer.py)
  * Parametric action implementation: [ContinuousActionDQNTrainer](https://github.com/caffe2/reinforcement-learning-models/blob/master/ml/rl/training/continuous_action_dqn_trainer.py)

# Contact us

If there are any issues/feedback with the implementations, feel free to file an issue: https://github.com/caffe2/reinforcement-learning-models/issues

Otherwise feel free to contact jjg@fb.com or nishadsingh@fb.com with questions.

# License

rl_caffe2 is BSD-licensed. We also provide an additional patent grant.
