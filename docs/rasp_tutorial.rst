.. _rasp_tutorial:

ReAgent Serving Platform (RASP)
===============================

Welcome to the ReAgent Serving Platform! This tutorial gets readers
familiar with reasoning at scale by building an artificial e-commerce
site.

What is RASP?
-------------

RASP is a set of scoring and ranking functions and a systematic way to
collect data and deploy models. A set of potential actions are input to
RASP and a ranked list of scores is output. The method for scoring and
ranking actions is called a decision plan. In this tutorial, we will
create several different decision plans and simulate user traffic to see
the results.

Installing ReAgent
------------------

Before beginning this tutorial, please install ReAgent by following
these instructions:
https://github.com/facebookresearch/ReAgent/blob/master/docs/installation.rst

Store set-up
------------

For this tutorial, we will be in charge of recommendations for
Pig-E-Barbeque, a pork e-store. Pig-E-Barbeque sells two products: Ribs
and Bacon. Our product manager has told us to optimize for clicks. Since
we are just starting out, we don’t know anything about our visitors, but
we know that bacon is delicious. We give bacon a score of 1.1 and ribs a
score of 1.0. (If we were optimizing for revenue, we could set the score
to the price, or we could have a custom scoring function.)

We also need to provide a ranking function that takes our scores and
decides which items to recommend. In Pig-E-Barbeque, we only have one
spot for recommendations, so the first item will be shown to visitors
and the second choice is discarded. If we use a greedy ranking function,
we will always show bacon (with it’s score of 1.1) and never show ribs
(with a score of 0.9). This means we will never know the true
performance of recommending ribs and can’t improve our system in the
future. This is known as the cold-start or explore-exploit problem
( https://arxiv.org/abs/1812.00116 ).

To avoid that problem, we will use the SoftmaxRanker, which will show
bacon 52% of the time and ribs 48% of the time. The SoftmaxRanker
operator is based on the softmax function:

::

   >>> import numpy as np
   >>>
   >>> def softmax(x):
   ...     e_x = np.exp(x - np.max(x))
   ...     return e_x / e_x.sum()
   ...
   >>> print(softmax([1.1, 1.0]))
   [0.52497919 0.47502081]

Here is the decision plan generator:

::

   def softmaxranker_decision_plan():
       op = SoftmaxRanker(temperature=1.0, values={"Bacon": 1.1, "Ribs": 1.0})
       return DecisionPlanBuilder().set_root(op).build()

And here is the generated decision plan:

::

   {
       "operators": [
           {
               "name": "SoftmaxRanker_1",
               "op_name": "SoftmaxRanker",
               "input_dep_map": {
                   "temperature": "constant_2",
                   "values": "constant_3"
               }
           }
       ],
       "constants": [
           {
               "name": "constant_2",
               "value": {
                   "double_value": 1.0
               }
           },
           {
               "name": "constant_3",
               "value": {
                   "map_double_value": {
                       "Bacon": 1.1,
                       "Ribs": 1.0
                   }
               }
           }
       ],
       "num_actions_to_choose": 1,
       "reward_function": "reward",
       "reward_aggregator": "sum"
   }

User simulator
--------------

Because this isn’t a real store, we need a way to simulate users. Our
simulator has a few rules:

1. Visitors click on bacon recommendations 50% of the time
2. 10% of visits are by rib lovers and the rest are regular visitors

   1. Rib lovers click on rib recommendations 90% of the time
   2. Regular visitors click on rib recommendations 10%

We will be using the built-in web service directly for this tutorial.
The simulator code can be found at:
serving/examples/ecommerce/customer_simulator.py

Makin’ bacon
------------

In one terminal window, start the RP server:

::

   ➜  ./serving/build/RaspCli --logtostderr
   I1014 17:23:19.736086 457250240 DiskConfigProvider.cpp:10] READING CONFIGS FROM serving/examples/ecommerce/plans
   I1014 17:23:19.738142 457250240 DiskConfigProvider.cpp:42] GOT CONFIG multi_armed_bandit.json AT serving/examples/ecommerce/plans/multi_armed_bandit.json
   I1014 17:23:19.738286 457250240 DiskConfigProvider.cpp:46] Registered decision config: multi_armed_bandit.json
   I1014 17:23:19.738932 457250240 DiskConfigProvider.cpp:42] GOT CONFIG contextual_bandit.json AT serving/examples/ecommerce/plans/contextual_bandit.json
   I1014 17:23:19.739020 457250240 DiskConfigProvider.cpp:46] Registered decision config: contextual_bandit.json
   I1014 17:23:19.739610 457250240 DiskConfigProvider.cpp:42] GOT CONFIG heuristic.json AT serving/examples/ecommerce/plans/heuristic.json
   I1014 17:23:19.739682 457250240 DiskConfigProvider.cpp:46] Registered decision config: heuristic.json
   I1014 17:23:19.739843 131715072 Server.cpp:58] STARTING SERVER

Then in another, run our simulator. The simulator will spawn many
threads and call RASP 1,000 times:

::

   ➜  python serving/examples/ecommerce/customer_simulator.py heuristic.json
   0
   200
   100
   400
   300
   500
   600
   700
   800
   900
   Average reward: 0.363
   Action Distribution: {'Ribs': 471, 'Bacon': 529}

As expected, we recommend Bacon 52% of the time and Ribs 48% of the
time. We get an average reward (in this case, average # of clicks) of about 0.36.

This is our baseline performance, but can we do better? From the log, we
can see that more bacon recommendations were clicked on:

::

   ➜  cat /tmp/rasp_logging/log.txt | grep '"name":"Ribs"}]' | grep '"reward":0.0' | wc -l
       390 # Ribs not clicked
   ➜  cat /tmp/rasp_logging/log.txt | grep '"name":"Ribs"}]' | grep '"reward":1.0' | wc -l
        88 # Ribs clicked
   ➜  cat /tmp/rasp_logging/log.txt | grep '"name":"Bacon"}]' | grep '"reward":1.0' | wc -l
       266 # Bacon clicked
   ➜  cat /tmp/rasp_logging/log.txt | grep '"name":"Bacon"}]' | grep '"reward":0.0' | wc -l
       253 # Bacon not clicked

This makes sense since, from our simulator definition, most people
aren’t rib-lovers and only click on ribs 10% of the time. We can change
the decision plan to use a multi-armed bandit that will learn to show
bacon much more often. For this tutorial, we will use the UCB1 bandit
ranker. Passing this to the plan generator:

::

   def ucb_decision_plan():
       op = UCB(method="UCB1", batch_size=16)
       return DecisionPlanBuilder().set_root(op).build()

Generates this plan:

::

   ➜  cat serving/examples/ecommerce/plans/multi_armed_bandit.json
   {
       "operators": [
           {
               "name": "UCB_1",
               "op_name": "Ucb",
               "input_dep_map": {
                   "method": "constant_2",
                   "batch_size": "constant_3"
               }
           }
       ],
       "constants": [
           {
               "name": "constant_2",
               "value": {
                   "string_value": "UCB1"
               }
           },
           {
               "name": "constant_3",
               "value": {
                   "int_value": 16
               }
           }
       ],
       "num_actions_to_choose": 1,
       "reward_function": "reward",
       "reward_aggregator": "sum"
   }

Running with this new plan gives:

::

   ➜  python serving/examples/ecommerce/customer_simulator.py multi_armed_bandit.json
   0
   200
   100
   400
   300
   500
   600
   700
   800
   900
   Average reward: 0.447
   Action Distribution: {'Ribs': 184, 'Bacon': 816}

This is already better than our previous score of 0.363. While we were
running, the bandit was learning and adapting the scores. Let’s run
again:

::

   ➜  python serving/examples/ecommerce/customer_simulator.py multi_armed_bandit.json
   0
   200
   100
   400
   300
   500
   600
   700
   800
   900
   Average reward: 0.497
   Action Distribution: {'Bacon': 926, 'Ribs': 74}

So the new ranker chooses bacon more often and gets more reward on
average than our first plan. If we keep running, eventually the model
will stop exploring the Ribs action and the average reward will approach
50% (which is the chance of a reward that we set in our simulator).

Straight Outta Context
----------------------

While running the store, our data scientist has discovered a way to
figure out who is a rib-lover. Now we can pass a context feature which
is 1 when the visitor is a rib lover and 0 otherwise. In this section we
will train a contextual bandit that learns to show ribs to rib lovers
and bacon to everyone else.

As we specified in our config, RP has been writing a log of visits and
feedback to a file. We can input this file with a training config to
ReAgent to train a contextual bandit model. First, let’s clear our
training data and start over by sending a SIGINT (control-c) to our
instance of RaspCli:

::

   …
   I1014 17:45:36.613893 6602752 Server.cpp:58] STARTING SERVER
   ^C
   ➜  rm /tmp/rasp_logging/log.txt
   ➜  ./serving/build/RaspCli --logtostderr
   I1014 17:48:49.674149 144418240 DiskConfigProvider.cpp:10] READING CONFIGS FROM serving/examples/ecommerce/plans
   I1014 17:48:49.678155 144418240 DiskConfigProvider.cpp:42] GOT CONFIG multi_armed_bandit.json AT serving/examples/ecommerce/plans/multi_armed_bandit.json
   I1014 17:48:49.679606 144418240 DiskConfigProvider.cpp:46] Registered decision config: multi_armed_bandit.json
   I1014 17:48:49.680496 144418240 DiskConfigProvider.cpp:42] GOT CONFIG contextual_bandit.json AT serving/examples/ecommerce/plans/contextual_bandit.json
   I1014 17:48:49.680778 144418240 DiskConfigProvider.cpp:46] Registered decision config: contextual_bandit.json
   I1014 17:48:49.682201 144418240 DiskConfigProvider.cpp:42] GOT CONFIG heuristic.json AT serving/examples/ecommerce/plans/heuristic.json
   I1014 17:48:49.682344 144418240 DiskConfigProvider.cpp:46] Registered decision config: heuristic.json
   I1014 17:48:49.682667 65638400 Server.cpp:58] STARTING SERVER

Now let’s run the heuristic model a few times to generate enough data
(this may take a few minutes). At the end there should be 10000 samples
(we can verify this with the wc command):

::

   ➜  for run in {1..10}; do python serving/examples/ecommerce/customer_simulator.py heuristic.json; done
   0
   200
   ...
   900
   Average reward: 0.36
   Action Distribution: {'Bacon': 516, 'Ribs': 484}
   ➜  wc -l /tmp/rasp_logging/log.txt
      10000 /tmp/rasp_logging/log.txt

RASP’s logging format and the ReAgent models’ input format is slightly
different. Fortunately, there’s a tool to convert from one to the other:

::

   ➜  python serving/scripts/rasp_to_model.py /tmp/rasp_logging/log.txt ecom_cb_input_data/input.json
   ➜  wc -l ecom_cb_input_data/input.json
      10000 ecom_cb_input_data/input.json

Since we are using the contextual bandit or RL model, we need to build a
timeline:

::

   rm -Rf spark-warehouse derby.log metastore_db preprocessing/spark-warehouse preprocessing/metastore_db preprocessing/derby.log ; /usr/local/spark/bin/spark-submit \
     --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \
     "`cat serving/examples/ecommerce/training/timeline.json`"
   ...
   2019-10-14 19:04:18 INFO  ShutdownHookManager:54 - Shutdown hook called
   2019-10-14 19:04:18 INFO  ShutdownHookManager:54 - Deleting directory /private/var/folders/jm/snmq7xfn7llc1tpnjgn7889h6l6pkw/T/spark-2b6a4171-cb60-4d5e-8052-87620a0677a2
   2019-10-14 19:04:18 INFO  ShutdownHookManager:54 - Deleting directory /private/var/folders/jm/snmq7xfn7llc1tpnjgn7889h6l6pkw/T/spark-927dae4a-6613-4a28-9d88-4d43a03d1cf3
   ➜  

The spark job creates a directory full of files, so we must merge into
one file for training & evaluation:

::

   ➜  mkdir -p training_data
   ➜  cat ecom_cb_training/part* > training_data/train.json
   ➜  cat ecom_cb_eval/part* > training_data/eval.json

Now we run our normalization. Any time we use a deep neural network, we
need normalization to prevent some large features from drowning others.

::

   ➜  python reagent/workflow/create_normalization_metadata.py -p serving/examples/ecommerce/training/cb_train.json

   WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
   INFO:ml.rl.preprocessing.normalization:Got feature: 0
   INFO:ml.rl.preprocessing.normalization:Feature 0 normalization: NormalizationParameters(feature_type='BINARY', boxcox_lambda=None, boxcox_shift=0.0, mean=0.0, stddev=1.0, possible_values=None, quantiles=None, min_value=0.0, max_value=1.0)
   INFO:ml.rl.preprocessing.normalization:Got feature: 1
   INFO:ml.rl.preprocessing.normalization:Feature 1 normalization: NormalizationParameters(feature_type='BINARY', boxcox_lambda=None, boxcox_shift=0.0, mean=0.0, stddev=1.0, possible_values=None, quantiles=None, min_value=1.0, max_value=1.0)
   INFO:__main__:`state_features` normalization metadata written to training_data/state_features_norm.json

Now we can train our contextual bandit:

::

   ➜  rm -Rf "outputs/*" ; python reagent/workflow/dqn_workflow.py -p serving/examples/ecommerce/training/cb_train.json
   INFO:ml.rl.json_serialize:TYPE:
   INFO:ml.rl.json_serialize:{'gamma': 0.0, 'target_update_rate': 1.0, 'maxq_learning': True, 'epsilon': 0.2, 'temperature': 0.35, 'softmax_policy': 0}
   ...
   INFO:ml.rl.workflow.page_handler:CPE evaluation took 0.26366519927978516 seconds.
   INFO:ml.rl.workflow.base_workflow:Training finished. Processed ~6555 examples / s.
   INFO:ml.rl.preprocessing.preprocessor:CUDA availability: False
   INFO:ml.rl.preprocessing.preprocessor:NOT Using GPU: GPU not requested or not available.
   /Users/jjg/github/Horizon/reagent/preprocessing/preprocessor.py:546: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
     elif max_value.gt(MAX_FEATURE_VALUE):
   /Users/jjg/github/Horizon/reagent/preprocessing/preprocessor.py:552: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
     elif min_value.lt(MIN_FEATURE_VALUE):
   INFO:__main__:Saving PyTorch trainer to outputs/trainer_1571105504.pt
   INFO:ml.rl.workflow.base_workflow:Saving TorchScript predictor to outputs/model_1571105504.torchscript

At this point, we have a model in ``outputs/model_*.torchscript``. We
are going to combine this scoring model with an Softmax ranker. The
ranker chooses the best actions most of the time, but rarely
chooses other actions to explore:

::

   {
       "operators": [
           {
               "name": "ActionValueScoringOp",
               "op_name": "ActionValueScoring",
               "input_dep_map": {
                   "model_id": "model_id",
                   "snapshot_id": "snapshot_id"
               }
           },
           {
               "name": "SoftmaxRankerOp",
               "op_name": "SoftmaxRanker",
               "input_dep_map": {
                   "temperature": "constant_2",
                   "values": "ActionValueScoringOp"
               }
           }
       ],
       "constants": [
           {
               "name": "model_id",
               "value": {
                   "int_value": 0
               }
           },
           {
               "name": "snapshot_id",
               "value": {
                   "int_value": 0
               }
           },
           {
               "name": "constant_2",
               "value": {
                   "double_value": 0.001
               }
           }
       ],
       "num_actions_to_choose": 1,
       "reward_function": "reward",
       "reward_aggregator": "sum"
   }

The “model_id” and “snapshot_id” tell us where to find the model. Let’s
put the model there so we can find it:

::

   ➜  mkdir -p /tmp/0
   ➜  cp outputs/model_*.torchscript /tmp/0/0

Let’s run with our model:

::

   ➜  python serving/examples/ecommerce/customer_simulator.py contextual_bandit.json
   0
   200
   100
   400
   300
   500
   600
   700
   800
   900
   Average reward: 0.52
   Action Distribution: {'Bacon': 883, 'Ribs': 117}

Nice! We have a reward higher than 50%, which is the click-through-rate
for bacon. This means that we must be getting most of the rib lovers. In
case you were curious, the best possible score is (0.9*0.5 + 0.1*\ 0.9)
== 0.54. We still have some exploration in our new plan so we won’t get
exactly 0.54 even with many iterations, but we need that exploration to
generate an even better model next time when we learn more about our
customers.

All of the decisions made so far have been pointwise: we don’t consider
repeat visitors. ReAgent can also optimize for long-term value in
sequential decisions using reinforcement learning, but that is out of
the scope of this starting tutorial.
