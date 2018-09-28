![Alt text](logo/horizon_banner.png)
### Applied Reinforcement Learning @ Facebook
[![Build Status](https://ci.pytorch.org/jenkins/buildStatus/icon?job=horizon-master)](https://ci.pytorch.org/jenkins/job/horizon-master/)
---

#### Overview
<TODO: add stuff from dex here>

#### Installation

##### Docker

We have included a Dockerfile for the CPU-only build and CUDA build under the docker directory.
The CUDA build will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run.

To build, `cd` into the respective directory and run

```
docker build -t horizon:dev .
```

If the Horizon unittests seem stuck, your Docker VM might not have enough memory.
In that case, multiprocessing might be killed and the tests could be left in a hanging state.
You can try to [increase docker resource limit](https://docs.docker.com/config/containers/resource_constraints/)
or comment out `python3 setup.py test` from Dockerfile.

##### Linux (Ubuntu)

Clone repo:
```
git clone https://github.com/facebookresearch/Horizon.git
cd Horizon/
```

Our project uses Thrift to define configuration and Spark to transform training data into the right format.
They require installing dependencies not managed by virtualenv. Here is the list of software needed to be installed on your system.
- Thrift compiler version 0.11.0 or above. You will need to build from source.
  See [1](https://thrift.apache.org/docs/install/debian), [2](https://thrift.apache.org/docs/BuildingFromSource).
- [Oracle Java 8](https://launchpad.net/~webupd8team/+archive/ubuntu/java)
- Maven

To install them all, you can run `./install_compilers.sh`. After it finished, you will need to add this line to your `.bash_profile`

```
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
```

Now, we recommend you to create virtualenv so that python dependencies can be contained in this project.

```
virtualenv -p python3 env
. env/bin/activate
```

First, install dependencies:

```
pip install -r requirements.txt
```

Then, install appropriate PyTorch 1.0 nightly build into the virtual environment:
```
# For CPU build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# For CUDA 9.0 build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# For CUDA 9.2 build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
```

After that, you will need to generate the python code of the thrift config definition. If you changed the thrift later on, you will have to rerun this.

```
thrift --gen py --out . ml/rl/thrift/core.thrift
```

And now, you are ready for installation.

```
pip install -e .
```

At this point, you should be able to run all unit tests:

```
python setup.py test
```

#### Usage

##### Online RL Training
Horizon supports online training environments for model testing. To train a model on OpenAI Gym, simply run:
```
python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_v0.json
```
Configs for different environments and algorithms can be found in `ml/rl/test/gym/`.

##### Batch RL Training Details
Horizon also supports training on offline data (Batch RL).

##### Quick Batch RL Examples
Discrete-Action DQN Workflow
```
cp ml/rl/workflow/sample_datasets/discrete_action/cartpole_training_data.json.gz ~
cp ml/rl/workflow/sample_datasets/discrete_action/state_features_norm.json.gz ~
gunzip ~/cartpole_training_data.json.gz ~/state_features_norm.json.gz

python ml/rl/workflow/dqn_workflow.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json
```

Parametric-Action DQN Workflow
```
cp ml/rl/workflow/sample_datasets/parametric_action/cartpole_training_data.json.gz ~
cp ml/rl/workflow/sample_datasets/parametric_action/state_features_norm.json.gz ~
cp ml/rl/workflow/sample_datasets/parametric_action/action_norm.json.gz
gunzip ~/cartpole_training_data.json.gz ~/state_features_norm.json.gz ~/action_norm.json.gz

python ml/rl/workflow/parametric_dqn_workflow.py -p ml/rl/workflow/sample_configs/parametric_action/parametric_dqn_example.json
```

DDPG Workflow
```
cp ml/rl/workflow/sample_datasets/continuous_action/pendulum_training_data.json.gz ~
cp ml/rl/workflow/sample_datasets/continuous_action/state_features_norm.json.gz ~
cp ml/rl/workflow/sample_datasets/continuous_action/action_norm.json.gz ~
gunzip ~/pendulum_training_data.json.gz ~/state_features_norm.json.gz ~/action_norm.json.gz

python ml/rl/workflow/ddpg_workflow.py -p ml/rl/workflow/sample_configs/continuous_action/ddpg_example.json
```

##### Detailed Overview
For DQN training, we expect the input data to have the following schema:

<TODO: add schema>

An example data set with this schema is given in ml/rl/workflow/sample_datasets/discrete_action/cartpole_pre_timeline.json.gz.

To train a DQN model on this data set we do the following:

Copy and unzip example dataset:

```
mkdir cartpole_discrete
cp ml/rl/workflow/sample_datasets/discrete_action/cartpole_pre_timeline.json.gz cartpole_discrete/
gunzip cartpole_discrete/cartpole_pre_timeline.json.gz
```
Models are trained on consecutive pairs of state/action tuples. To assist in creating this table, we have an `RLTimelineOperator` spark operator. Build and run the timeline operator on the data. Make sure that you have java (not openjdk that sometimes is shipped with linux) & scala installed:
```
mvn -f preprocessing/pom.xml package
```
Next, download spark if you don't already have it:
```
wget http://www-eu.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
tar xvf spark-2.3.1-bin-hadoop2.7.tgz
mv spark-2.3.1-bin-hadoop2.7 /usr/local/spark
```
Now run the timeline operator on the training data directory to generate the training data:
```
/usr/local/spark/bin/spark-submit --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar  "cat ml/rl/workflow/sample_configs/discrete_action/timeline.json"
```
This will create a directory `cartpole_discrete_training_data` that contains the (sharded) post timeline data. An example of this data is given in ml/rl/workflow/sample_datasets/discrete_action/cartpole_training_data.json.gz. We will use this example to train our DQN model.

```
cp ml/rl/workflow/sample_datasets/discrete_action/cartpole_training_data.json.gz ~/
gunzip ~/cartpole_training_data.json.gz
```
Next, we will create our normalization meta-data based off of the features in the training data. We run a one-time normalization workflow that analyzes the dataset and determines the best normalization parameters.
```
python ml/rl/workflow/create_normalization_metadata.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json
```
This will create a file that contains our feature normalization meta-data at the path specified in `dqn_example.json`. Next we can run the DQN training workflow as follows:
```
python ml/rl/workflow/dqn_workflow.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json
```
This command trains the DQN model on the training data `~/cartpole_training_data` using the normalization parameters `~/state_features_norm.json`.

Upon completion of training two models are ouput to file. We output a snaphot of the PyTorch trainer object - a python object that holds all objects necessary to resume training (neural nets, optimizers, etc.) and a caffe2 model which can be used in production for inference across many devices. See `test_read_c2_model_from_file` in `ml/rl/test/workflow/test_oss_workflows.py ` for an example of how to use the outputted caffe2 model in Python.
