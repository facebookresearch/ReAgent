# BlueWhale (Applied Reinforcement Learning @ Facebook)

![BlueWhale Logo](https://raw.githubusercontent.com/facebookresearch/BlueWhale/master/logo/cover_image_light.png)

How would you teach a robot to balance a pole? Or safely land a space ship? Or even to walk?

Using reinforcement learning (RL), you wouldn't have to teach it how to do any of these things: only what to do. RL formalizes our intuitions about trial and error – agents take actions, experience feedback, and adjust their behavior accordingly.

An agent may start with awful performance: the cart drops the pole immediately; when the space ship careens left, it tilts further; the walker can't take one step without falling. But with experience from exploration and failure, it learns. Soon enough, the agent is behaving in a way you never explicitly told it to, and is achieving the goals you implicitly set forth. It takes actions that optimize for the reward system you designed, often coming up with solutions and employing strategies you hadn't thought of.

While historically, RL has been primarily used in the context of robotics and game-playing, it can be employed in a variety of problem spaces. At Facebook, we're working on using RL at scale: suggesting people you may know, notifying you about fiend & page updates, optimizing our streaming video bitrate, and more.

Advances in RL theory, including the advent of Deep Query Networks and Deep Actor-Critic models, allow us to use function approximation to approach problems with large state and action spaces.  This project, called BlueWhale, contains Deep RL implementations built on [PyTorch](http://pytorch.org/) and [caffe2](http://caffe2.ai/). Internally, we train these models on large databases of episodes, but externally we provide support for running BlueWhale inside [OpenAI Gym](gym.openai.com).  BlueWhale is extremely fast and can train models with 10M+ parameters on billions of rows of data.

# Requirements

### Python3 (recommended for mac: Anaconda)

All of BlueWhale code depends on Python 3's type inference system.  For mac users, we recommend using [Anaconda](https://www.continuum.io/downloads) instead of the system implementation of python. Install anaconda and verify you are using Anaconda's version of python before installing other dependencies: `which python` should yield an Anaconda path.

### Caffe2

BlueWhale runs on any platform that supports caffe2. To install caffe2, follow this tutorial: [Installing Caffe2](https://caffe2.ai/docs/getting-started.html).

You may need to override caffe2's cmake defaults to use homebrew's protoc instead of Anaconda's protoc and to use Anaconda's Python instead of system Python.  Also add the following switch when running cmake to make sure caffe2 uses python3:

```
cmake -DPYTHON_EXECUTABLE=`which python3`
```

Also make sure cmake is using the **homebrew** version of glog, etc..  Sometimes caffe2
will try to use the anaconda version of these libraries, and that will cause errors.

### FBThrift

[FBThrift](https://github.com/facebookresearch/fbthrift) is Facebook's RPC framework.  Note that
we require *FBThrift*, not Apache Thrift.  Here are instructions for getting on OS/X

```
# Install deps with homebrew
brew install openssl zstd folly

# Wangle isn't in homebrew and needs to be installed manually
git clone https://github.com/facebookresearch/wangle.git
cd wangle/build
cmake ../ -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl
make -j8
make install
cd ../..

# Install FBThrift
git clone https://github.com/facebookresearch/fbthrift
cd fbthrift
mkdir build
cd build
cmake ../ -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl
make -j8
make install

# Install python bindings for FBThrift (There may be some compile errors, they can
#   be ignored)
cd ../thrift/lib/py
python setup.py install
```

### OpenAI Gym

Running models in OpenAI Gym environments requires platforms with OpenAI Gym support. Windows support for OpenAI Gym is being tracked [here](https://github.com/openai/gym/issues/11).

OpenAI Gym can be installed using [pip](https://pypi.python.org/pypi/pip) which should come with your python installation in the case of linux or with anaconda in the case of OSX.  To install the basic environments ([classic control](https://gym.openai.com/envs#classic_control), [toy text](https://gym.openai.com/envs#toy_text), and [algorithmic](https://gym.openai.com/envs#algorithmic)), run:

```
pip install gym
```

To install [all environments](https://gym.openai.com/envs/), run this instead:

```
pip install "gym[all]"
```

# Installation and Setup

Clone from source:

```
git clone https://github.com/facebookresearch/BlueWhale
cd BlueWhale
```

Create thrift generated code within the root directory:

```
thrift1 --gen py:json --out . ml/rl/thrift/core.thrift
```

To access caffe2 and import our modules:

```
export PYTHONPATH=/usr/local:./:$PYTHONPATH
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

You can run RL models of your specification on OpenAI Gym environments of your choice. Right now, we only support environments that supply `Box(x, )` or `Box(x, y, z)` (image) state representations and require `Discrete(y)` action inputs.

### Quickstart

```
python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/maxq_cartpole_v0.json
```

The [run_gym.py](https://github.com/facebookresearch/BlueWhale/tree/master/ml/rl/test/gym/run_gym.py) script will construct an RL model and run it in an OpenAI Gym environemnt, periodically reporting scores averaged over several trials. In general, you can run RL models in OpenAI Gym environments with:

```
python ml/rl/test/gym/run_gym.py -p <parameters_file> [-s <score_bar>] [-g <gpu_id>]
```
* **parameters_file**: Path to your JSON parameters file
* **score_bar** (optional): Scalar score you hope to achieve. Once your model scores at least *score_bar* well averaged over 100 test trials, training will stop and the script will exit. If left empty, training will continue until you complete collect data from *num_episodes* episodes (see details on parameters in the next section)
* **gpu_id** (optional): If set to your machine's GPU id (typically `0`), the model will run its training and inference on your GPU. Otherwise it will use your CPU

Feel free to create your own parameter files to select different environments and change model parameters. The success criteria for different environments can be found [here](https://gym.openai.com/envs). We currently supply default arguments for the following environments:

* [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) environment: [maxq\_cartpole\_v0.json](https://github.com/facebookresearch/BlueWhale/tree/master/ml/rl/test/gym/maxq_cartpole_v0.json)
* [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment: [maxq\_cartpole\_v1.json](https://github.com/facebookresearch/BlueWhale/tree/master/ml/rl/test/gym/maxq_cartpole_v1.json)
* [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) environment: [maxq\_lunarlander\_v2.json](https://github.com/facebookresearch/BlueWhale/tree/master/ml/rl/test/gym/maxq_lunarlander_v2.json)

Feel free to try out image-based environments too! The parameters we supply will get you a model that runs and trains quickly, not one that performs well:

* [Asteroids-v0](https://gym.openai.com/envs/Asteroids-v0/) environment: [maxq\_asteroids\_v0.json](https://github.com/facebookresearch/BlueWhale/tree/master/ml/rl/test/gym/maxq_asteroids_v0.json)

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
* **run_details** (reading the code that uses these might be helpful: [run\_gym.py](https://github.com/facebookresearch/BlueWhale/blob/master/ml/rl/test/gym/run_gym.py#L21))
  * **num_episodes**: Number of episodes run the mode and to collect new data over
  * **train_every**: Number of episodes between each training cycle
  * **train_after** Number of episodes after which to enable training
  * **test_every**: Number of episodes between each test cycle
  * **test_after**: Number of episodes after which to enable testing
  * **num\_train\_batches**: Number of batches to train over each training cycle
  * **avg\_over\_num\_episodes**: Number of episodes to run every test cycle. After each cycle, the script will report an average over the scores of the episodes run within it.The typical choice is `100`, but this should be set according to the [success criteria](https://gym.openai.com/envs) for your environment
  * **render**: Whether or not to render the OpenAI environment in training and testing episodes. Note that some environments don't support rendering
  * **render_every**: Number of episodes between each rendered episode



# Supported Models

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

# Contact Us

If you identify any issues or have feedback, please [file an issue](https://github.com/facebookresearch/BlueWhale/issues).

## Current Developers

Jason Gauci <jjg@fb.com>
Edoardo Conti <edoardoc@fb.com>

## Alumni

Nishad Singh
Yuchen He
Yannick Schröcker

# License

BlueWhale is BSD-licensed. We also provide an additional patent grant.
