.. _continuous_integration:

Continuous Integration
======================

We have CI setup on `CircleCI <https://circleci.com/gh/facebookresearch/ReAgent>`_.
It's a pretty basic setup. One key thing to note is that dependencies are baked into our Docker images.
If you need a new dependency, you will have to rebuild Docker images, upload them to Docker hub, and update ``.circleci/config.yml`` accordingly.
Here are the steps:

.. code-block::

   docker build -f docker/cpu.Dockerfile -t reagent:cpu .
   docker tag reagent:cpu kittipatv/reagent:cpu_test
   docker push kittipatv/reagent:cpu_test

If you cannot push, you have to run ``docker login``.

Then, you should follow the local testing instructions in ``.circleci/config.yml`` to make sure that the new Docker image is in a good shape.
Once you are sure, tag the image without ``_test`` suffix and push again.
If you weren't the last person to update the images, then you will have to submit a PR to update the image names.
