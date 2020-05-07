.. _usage:

Usage
=====

ReAgent is designed for large-scale, distributed recommendation/optimization tasks where we don't
have access to a simulator.  In this environment, it's typically better to train offline on batches
of data, and release new policies slowly over time.  Because the policy updates slowly and in
batches, we use *off-policy* algorithms.  To test a new policy without deploying it, we rely on
*counter-factual policy evaluation (CPE)*\ , a set of techniques for estimating a policy based on the
actions of another policy.

Quick Start
-----------

We have set up `Click <https://click.palletsprojects.com/en/7.x/>`_ commands to run our RL workflow. The basic usage pattern is

.. code-block::

    ./reagent/workflow/cli.py run <module.function> <path/to/config>


To train a model online with OpenAI Gym, simply run the Click command:

.. code-block::

    # set the config
    export CONFIG=reagent/gym/tests/configs/cartpole/discrete_dqn_cartpole_online.yaml
    # train and evaluate model on gym environment
   ./reagent/workflow/cli.py run reagent.gym.tests.test_gym.run_test $CONFIG


To train a batch RL model, run the following commands:

.. code-block::

    # set the config
    export CONFIG=reagent/workflow/sample_configs/discrete_dqn_cartpole_offline.yaml
    # gather some random transitions (can replace with your own)
    ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.offline_gym $CONFIG
    # convert data to timeline format
    ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.timeline_operator $CONFIG
    # train model based on timeline data, and evaluate
    ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.train_and_evaluate_gym $CONFIG


Now we will describe how the above commands work, starting with a traditional RL setup with a simulator where we can trivially evaluate new policies:

1 - On-Policy RL Training
-------------------------

OpenAI Gym is a set of environments: simulators that can run policies for a given task and generate rewards.  If a simulator is accessible, on-policy training (where the latest version of the policy makes new decisions in real-time) can give better results. We have a suite of benchmarks on OpenAI Gym, which is listed in ``reagent/gym/tests/test_gym.py``'s ``GYM_TESTS``. To train a model on OpenAI Gym, simply run the Click command:

.. code-block::

   ./reagent/workflow/cli.py run reagent.gym.tests.test_gym.run_test reagent/gym/tests/configs/cartpole/discrete_dqn_cartpole_online.yaml

Configs for different environments and algorithms can be found in ``reagent/gym/tests/configs/<env_name>/<algorithm>_<env_name>_online.yaml``.

While this is typically the set up for people conducting RL research, it isn't always practical to deploy on-policy RL for several reasons:


#. We don't have a simulator and the problem may be so complex that building an accurate simulator is non-trivial.
#. Thousands or even tens-of-thousands of machines must execute the policy in parallel, and keeping the latest policy in sync on all of these nodes is difficult
#. We want to evaluate the behavior of the policy offline and then keep the policy constant afterwards to reduce the risk that the policy will degrade at odd hours.
#. We are building on top of traditional recommender systems that typically rely on a fixed, stochastic policy.

For these reasons, ReAgent is designed to support batch, off-policy RL.  Let's now walk though how to train a model with ReAgent:

2- Offline RL Training (Batch RL)
---------------------------------

The main use case of ReAgent is to train RL models in the **batch** setting. In batch reinforcement learning the data collection and policy learning steps are decoupled. Specifically, we try to learn the best possible policy given the input data. In batch RL, being able to handle thousands of varying feature types and distributions and algorithm performance estimates before deployment are of key importance.

In this example, we will train a DQN model on Offline ``CartPole-v0`` data, where Click command config should be set to

.. code-block::

    export CONFIG=reagent/workflow/sample_configs/discrete_dqn_cartpole_offline.yaml


We now proceed to give pseudo-code to sketch out the main ideas of our batch RL workflow.


Step 1 - Create training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first generate data from a random policy (chooses random actions) run on the ``CartPole-v0`` environment. 
In particular, the following Click command runs 150 episodes of ``CartPole-v0`` (max steps of 200) and stored the pickled dataframe in ``/tmp/tmp_pickle.pkl``, which you may inspect via ``pd_df = pd.read_pickle(pkl_path)``.

.. code-block::

    ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.offline_gym $CONFIG

The command essentially performs the following pseudo-code:

.. code-block::

    dataset = RLDataset()
    for epoch in range(num_episodes_for_data_batch):
      run_episode & store transitions to dataset

    df = dataset.to_pandas_df()
    df.to_pickle(pkl_path)

In practice, end users would generate a dataset in a similar format from their production system. For this example, the data is stored as a pickled Pandas dataframe. 

This is human-readable, but not the most efficient way to store tabular data.  Other ways to store input data are parquet, CSV, or any other format that can be read by Apache Spark.  All of these formats are fine, as long as the following schema is maintained:

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


Once you have data on this format, you can move on to Step 2.

Step 2 - Convert the data to the ``Timeline`` format
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

Now that we are ready, let's run our spark job on our local machine. This will produce a massive amount of logging (because we are running many systems that typically are distributed across many nodes) and there will be some exception stack traces printed because we are running in a psuedo-distributed mode.  Generally this is fine as long as the output data is generated. To do so, run the following Click command:

.. code-block::
    
    ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.timeline_operator $CONFIG


The command essentially performs the following pseudo-code:

.. code-block::

   # load pandas dataframe
   pd_df = pd.read_pickle(pkl_path)

   # convert to Spark dataframe
   spark = get_spark_session()
   df = spark.createDataFrame(pd_df)

   # run timelime operator
   json_params = make_input_to_timeline_operator()
   spark._jvm.com.facebook.spark.rl.Timeline.main(json_params)


Now that our data is a Spark table in Hive storage, we're ready to run the training workflow (Steps 3-5). These steps are altogether accomplished with the following Click command:

.. code-block::

    ./reagent/workflow/cli.py run reagent.workflow.gym_batch_rl.train_and_evaluate_gym $CONFIG


We now proceed to describing this command and present some pseudo-code.


Step 3 - Determine normalization parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data from production systems is often sparse, noisy and arbitrarily distributed. Literature has shown that neural networks learn faster and better when operating on batches of features that are normally distributed. ReAgent includes a workflow that automatically analyzes the training dataset and determines the best transformation function and corresponding normalization parameters for each feature. We do this via ``ModelManager.run_feature_identification``, where ``input_table_spec`` points to a Spark table with the timeline data.

.. code-block::

   model: ModelManager__Union
   manager = model.value
   manager.run_feature_identification(input_table_spec)


The normalization is a Python dictionary where each key is a feature id and each value is NormalizationData.
An example of this, in JSON format, is

.. code-block::

   {
       "0": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.5675003528594971,\"stddev\":1.0,\"min_value\":-0.1467551738023758,\"max_value\":2.1779561042785645}",
       "1": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.42259514331817627,\"stddev\":1.0,\"min_value\":-1.3586808443069458,\"max_value\":1.8529225587844849}",
       "2": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.028220390900969505,\"stddev\":1.0,\"min_value\":-0.14581388235092163,\"max_value\":0.19483095407485962}",
       "3": "{\"feature_type\":\"CONTINUOUS\",\"mean\":0.02947876788675785,\"stddev\":1.0,\"min_value\":-2.194336175918579,\"max_value\":2.164193868637085}"
   }

NB: ``reagent/workflow/training.py`` is what the pseudo-code in Steps 3 and 4 are trying to depict. Models should subclass ``ModelManager`` and implement all abstract methods (including ``run_feature_identification`` and ``query_data``) to be added to our registry of models.

Step 4 - Train model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train the model, we first save our Spark table to Parquet format, and use `Petastorm <https://github.com/uber/petastorm>`_'s PyTorch DataLoader, which can efficiently read Parquet formatted data. We do this via ``ModelManager.query_data``, which each ``ModelManager`` in our registry of models must implement. In this step, we also process the rewards, i.e. computing multi-step rewards or computing the reward from ``metrics`` columns directly.

.. code-block::
    train_dataset = manager.query_data(
        input_table_spec=input_table_spec,  # description of Spark table
        sample_range=train_sample_range,  # what percentage of data to use for training
        reward_options=reward_options,  # config to calculate rewards
    )
    # train_dataset now points to a Parquet

Now we are ready to train a model by running:

.. code-block::

    # make preprocessor from the normalization parameters of Step 3
    batch_preprocessor = manager.build_batch_preprocessor()

    # read preprocessed data
    data_reader = petastorm.make_batch_reader(train_dataset.parquet_url)
    with DataLoader(data_reader, batch_preprocessor) as dataloader:
      for batch in dataloader:
        trainer.train(batch)


    # Store model outputs
    torchscript_output_path = f"model_{round(time.time())}.torchscript"
    serving_module = manager.build_serving_module()
    torch.jit.save(serving_module, torchscript_output_path)

    # store for later use
    training_output.output_path = torchscript_output_path

Note that the model is trained purely on the randomly generated data we collected in Step 1.
We are taking a batch of data that we generated previously and training by looping over that data and interatively learning a better policy than the policy that generated the data.
Effectively, this is learning to perform a task by observing completely random transitions from an environment! While doing so, we are not even building a dynamics model of the environment.

NB: We can do the same for the ``eval_dataset`` if we want to perform CPE during training as a diagnosis tool.

Step 5 - Evaluate the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have trained a new policy on the offline ``CartPole-v0`` data, we can try it out to see how it does:

.. code-block::

    # load our previous serving module
    jit_model = torch.jit.load(training_output.output_path)

    # wrap around module to fit our gymrunner interface
    policy = create_predictor_policy_from_model(env, jit_model)
    agent = Agent.create_for_env(
        env, policy=policy, action_extractor=policy.get_action_extractor()
    )

    # observe rewards
    for _ in range(num_eval_episodes):
        ep_reward = run_episode(env=env, agent=agent, max_steps=max_steps)


Even on completely random data, DQN was able to learn a policy that can obtain scores close to the maximum possible score of 200.


Step 6 - Visualize Results via Tensorboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now view loss plots and CPE estimates in Tensorboard after running:

.. code-block::

   tensorboard --logdir outputs/

at `localhost:6006  <localhost:6006>`_. When done viewing the results deactivate the virtualenv by typing ``deactivate``.
