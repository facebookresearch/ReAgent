.. _distributed:

Training a model across multiple GPUs
=====================================

Before we get started, please check out the :ref:`Usage Guide <usage>` and
the `PyTorch Distributed documentation <https://pytorch.org/docs/stable/distributed.html>`_.

How distributed training works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With a single GPU and model, training follows this process:

1. Compute the loss from a minibatch of data (the forward pass of the model)
2. Backpropagate that loss through the model to compute gradients (the backward pass of the model)
3. Sum the gradients across the minibatch.
4. Run the optimizer by calling the "step()" function.

Now assume we have several GPUs, and they each have exactly the same model.
#1 and #2 are `embarrassingly parallel <https://en.wikipedia.org/wiki/Embarrassingly_parallel>`_ and can be distributed to many nodes.
As long as we can sum across nodes to complete #3 (this is known as an 'all-reduce'), then each node can run #4 on the same gradients,
and the resulting models will again be identical.  This is the premise behind distributed training.

Training on a single node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using multiple GPUs on a single node is relatively straightforward.  When running either the dqn_workflow or the parametric_dqn_workflow,
set the "use_all_avail_gpus" parameters in the input config (the json file) to true.  ReAgent will detect the number of available GPUs and
run on all of them without any additional effort.

Training on multiple nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-node training requires more setup.  Some prerequisites:

1. A networked filesystem (such as NFS) that all trainers can access
2. A local filesystem that is unique to each trainer

To set up the environment for *N* nodes, do the following:

1. Create an empty file on the networked filesystem which the *N* machines will use to communicate
2. Either split the dataset into *N* equal parts, **or** shuffle the dataset to create N copies.
3. Put one copy of the dataset onto each machine's local filesystem in exactly the same path.
4. Set the "use_all_avail_gpus" parameter to true as above
5. Also set the "num_nodes" parameter to *N*.
6. On each machine, run the workflow with the "--node_index=n" flag, where n is the index of that machine.
7. The machine with --node_index=0 will save the final model to the output path specified.
