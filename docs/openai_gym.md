---
id: openai_gym
title: Running OpenAI Gym Environments
sidebar_label: OpenAI Gym
---

You can run RL models of your specification on OpenAI Gym environments of your choice. Right now, we only support environments that supply `Box(x, )` or `Box(x, y, z)` (image) state representations and require `Discrete(y)` action inputs.

## Quickstart

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

## Modifying the parameters file

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
