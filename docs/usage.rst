.. _usage:

Usage
=====

ReAgent is designed for large-scale, distributed recommendation/optimization tasks where we don't
have access to a simulator.  In this environment, it's typically better to train offline on batches
of data, and release new policies slowly over time.  Because the policy updates slowly and in
batches, we use *off-policy* algorithms.  To test a new policy without deploying it, we rely on
*counter-factual policy evaluation (CPE)*\ , a set of techniques for estimating a policy based on the
actions of another policy.

Before we get started using ReAgent as it is intended, let's begin with a traditional RL setup with a simulator where we can trivially evaluate new policies:

1 - On-Policy RL Training
-------------------------

Open AI Gym is a set of environments: simulators that can run policies for a given task and generate rewards.  If a simulator is accessible, on-policy training (where the latest version of the policy makes new decisions in real-time) can give better results. To train a model on OpenAI Gym, simply run:

.. code-block::

   python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_v0.json

Configs for different environments and algorithms can be found in ``ml/rl/test/gym/``.

While this is typically the set up for people conducting RL research, it isn't always practical to deploy on-policy RL for several reasons:


#. We don't have a simulator and the problem may be so complex that building an accurate simulator is non-trivial.
#. Thousands or even tens-of-thousands of machines must execute the policy in parallel, and keeping the latest policy in sync on all of these nodes is difficult
#. We want to evaluate the behavior of the policy offline and then keep the policy constant afterwards to reduce the risk that the policy will degrade at odd hours.
#. We are building on top of traditional recommender systems that typically rely on a fixed, stochastic policy.

For these reasons, ReAgent is designed to support batch, off-policy RL.  Let's now walk though how to train a model with ReAgent:

2- Offline RL Training (Batch RL)
---------------------------------

The main use case of ReAgent is to train RL models in the **batch** setting. In batch reinforcement learning the data collection and policy learning steps are decoupled. Specifically, we try to learn the best possible policy given the input data. In batch RL, being able to handle thousands of varying feature types and distributions and algorithm performance estimates before deployment are of key importance.

In this example, we will train a DQN model on Offline ``Cartpole-v0`` data:

Step 1 - Create training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First we need to generate the data required to train our RL models. For this example we generate data from the ``Cartpole-v0`` environment in OpenAI Gym. In practice, end users would generate a dataset in a similar format from their production system.

.. code-block::

   mkdir cartpole_discrete

   python ml/rl/test/gym/run_gym.py -p ml/rl/test/gym/discrete_dqn_cartpole_small_v0.json -f cartpole_discrete/training_data.json --seed 0

Let's look at one row of data to see the expected input format:

.. code-block::

   cat cartpole_discrete/training_data.json | head -n1 | python -m json.tool

   {
       "mdp_id": "10",
       "sequence_number": 0,
       "state_features": {
           "2": 0.0021880944,
           "1": -0.015781501,
           "0": -0.031933542,
           "3": 0.04611974
       },
       "action": "0",
       "reward": 1.0,
       "possible_actions": [
           "0",
           "1"
       ],
       "action_probability": 0.9,
       "ds": "2018-06-25"
   }

The input data is a flat file containing a JSON object per-line separated by newlines (the first line is pretty-printed here for readability).  This is human-readable, but not the most efficient way to store tabular data.  Other ways to store input data is parquet, CSV, or any other format that can be read by Apache Spark.  All of these formats are fine, as long as the following schema is maintained:

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - mdp_id
     - string
     - A unique ID for the episode (e.g. an entire playthrough of a game)
   * - sequence_number
     - integer
     - Defines the ordering of states in an MDP (e.g. the timestamp of an event)
   * - state_features
     - ``map<integer,float>``
     - A set of features describing the state.
   * - action
     - string
     - The name of the action chosen
   * - reward
     - float
     - The reward at this state/action
   * - possible_actions
     - ``list<string>``
     - A list of all possible actions at this state.  Note that the action taken must be present in this list.
   * - action_probability
     - float
     - The probability of taking this action if the policy is stochastic, else ``null``.  Note that we strongly encourage using a stochastic policy instead of choosing the best action at every timestep.  This exploration will improve the evaluation and ultimately result in better learned policies.
   * - ds
     - string
     - A unique ID for this dataset.


Note that JSON does not support integer keys in objects so in our JSON format we replace the ``map<integer,float>`` with ``map<string,float>``\ , but even in this case the keys must be strings of integers.

Once you have data on this format (or you have generated data using our gym script) you can move on to step 2:

Step 2 - Convert the data to the ``timeline`` format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models are trained on consecutive pairs of state/action tuples. To assist in creating this table, we have an ``RLTimelineOperator`` spark operator. Let's build and run the timeline operator on the data:

First, we need to build the Spark library that will execute the timeline.  Apache Spark is a platform for doing massively-parallel processing.  Although we are running this on a single file, Spark is designed to work on thousands of files distribued across many machines.  Explaining HDFS, Hive, and Spark are beyond the scope of this tutorial, but for large datasets it's important to understand these concepts and that it's possible to run ReAgent in a distributed environment by simply changing the location of the input from a file to an HDFS folder.

.. code-block::

   # Build timeline package (only need to do this first time)
   mvn -f preprocessing/pom.xml clean package

When running spark locally, spark creates a fake "cluster" where it stores all of the data.  We want to remove this before running so we don't accidentally pull in data from a prior run.  In a production setting, we would delete the output data table before running using a Hive command.

.. code-block::

   # Clear last run's spark data (in case of interruption)
   rm -Rf spark-warehouse derby.log metastore_db preprocessing/spark-warehouse preprocessing/metastore_db preprocessing/derby.log

Now that we are ready, let's run our spark job on our local machine.  This will produce a massive amount of logging (because we are running many systems that typically are distributed across many nodes) and there will be some exception stack traces printed because we are running in a psuedo-distributed mode.  Generally this is fine as long as the output data is generated:

.. code-block::

   # Run timelime on pre-timeline data
   /usr/local/spark/bin/spark-submit \
     --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \
     "`cat ml/rl/workflow/sample_configs/discrete_action/timeline.json`"

   # Look at the first row of training & eval
   head -n1 cartpole_discrete_training/part*

   head -n1 cartpole_discrete_eval/part*

There are many output files.  The reason for this is that Spark expects many input & output files: otherwise it wouldn't be able to efficiently run on many machines and output data in parallel.  For this tutorial, we will merge all of this data into a single file, but in a production use-case we would be streaming data from HDFS during training.

.. code-block::

   # Merge output data to single file
   mkdir training_data
   cat cartpole_discrete_training/part* > training_data/cartpole_discrete_timeline.json
   cat cartpole_discrete_eval/part* > training_data/cartpole_discrete_timeline_eval.json

   # Remove the output data folder
   rm -Rf cartpole_discrete_training cartpole_discrete_eval

Now that all of our data has been grouped into consecutive pairs, we can run the normalization pipeline.

Step 3 - Create the normalization parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data from production systems is often sparse, noisy and arbitrarily distributed. Literature has shown that neural networks learn faster and better when operating on batches of features that are normally distributed. ReAgent includes a workflow that automatically analyzes the training dataset and determines the best transformation function and corresponding normalization parameters for each feature. We can run this workflow on the post timeline data:

.. code-block::

   python ml/rl/workflow/create_normalization_metadata.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json

Now we can look at the normalization file.  It's a JSON file where each key is a feature id and each value is a string-encoded JSON object describing the normalization:

.. code-block::

   cat training_data/state_features_norm.json | python -m json.tool

   {
       "0": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.5675003528594971,\"stddev\":1.0,\"min_value\":-0.1467551738023758,\"max_value\":2.1779561042785645}",
       "1": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.42259514331817627,\"stddev\":1.0,\"min_value\":-1.3586808443069458,\"max_value\":1.8529225587844849}",
       "2": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.028220390900969505,\"stddev\":1.0,\"min_value\":-0.14581388235092163,\"max_value\":0.19483095407485962}",
       "3": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.02947876788675785,\"stddev\":1.0,\"min_value\":-2.194336175918579,\"max_value\":2.164193868637085}"
   }

Step 4 - Train model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we are ready to train a model by running:

.. code-block::

   # Store model outputs here
   mkdir outputs

   python ml/rl/workflow/dqn_workflow.py -p ml/rl/workflow/sample_configs/discrete_action/dqn_example.json

Note that, even in the OpenAI Gym case, we aren't running the gym at this step.  We are taking a batch of data that we generated previously and training by looping over that data and interatively learning a better policy than the policy that generated the data.

Step 5 - Evaluate the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have trained a new policy on the offline ``Cartpole-v0`` data, we can try it out to see how it does:

.. code-block::

   python ml/rl/test/workflow/eval_cartpole.py -m outputs/model_* --softmax_temperature=0.35 --log_file=outputs/eval_output.txt

Step 6 - Visualize Results via Tensorboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now view loss plots and CPE estimates in Tensorboard after running:

.. code-block::

   tensorboard --logdir outputs/

at `localhost:6006  <localhost:6006>`_. When done viewing the results deactivate the virtualenv by typing ``deactivate``.
