# Usage

## 1 - Online RL Training
Horizon supports online training environments for model testing. To train a model on OpenAI Gym, simply run:
```
python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_v0.json
```
Configs for different environments and algorithms can be found in `ml/rl/test/gym/`.

## 2- Offline RL Training (Batch RL)

The main use case of Horizon is to train RL models in the **batch** setting. In batch reinforcement learning the data collection and policy learning steps are decoupled. Specifically, we try to learn the best possible policy given the input data. In batch RL, being able to handle thousands of varying feature types and distributions and algorithm performance estimates before deployment are of key importance.

#### End-to-End example: Training a DQN model on Offline `Cartpole-v0` data:

##### Step 1 - Create training data
First we need to generate the data required to train our RL models. For this example we generate data from the `Cartpole-v0` environment in OpenAI Gym. In practice, end users would generate a dataset in a similar format from their production system.

```
./scripts/gen_training_data.sh
```
Alternatively, to skip generating the Gym data, you can use the pre-generated data found in `ml/rl/workflow/sample_datasets`.
##### Step 2 - Convert the data to the `timeline` format
Models are trained on consecutive pairs of state/action tuples. To assist in creating this table, we have an `RLTimelineOperator` spark operator. Build and run the timeline operator on the data:
```
./scripts/run_timeline.sh
```
##### Step 3 - Create the normalization parameters
Data from production systems is often sparse, noisy and arbitrarily distributed. Literature has shown that neural networks learn faster and better when operating on batches of features that are normally distributed. Horizon includes a workflow that automatically analyzes the training dataset and determines the best transformation function and corresponding normalization parameters for each feature. We can run this workflow on the post timeline data:
```
./scripts/create_normalization.sh
```
##### Step 4 - Train model
Now we are ready to train a model by running:
```
./scripts/train.sh
```
##### Step 5 - Evaluate Model
Now that we have trained a new policy on the offline `Cartpole-v0` data, we can try it out to see how it does:
```
./scripts/eval.sh predictor_<number>.c2
```
