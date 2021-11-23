.. _installation:

Installation
============

ReAgent CLI & Python API
^^^^^^^^^^^^^^^^^^^^^^^^

We have CLI to launch training & Python API to use programmatically, e.g., in your own script or Jupyter Notebook.
To install this component, you will need to have Python 3.8+ installed on your system.
If you don't have that, you can either install it via `pyenv <https://github.com/pyenv/pyenv>`_ or
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_. To verify that you have the right version,
type the following command on your shell:

.. code-block:: bash

   python --version

Once you make sure you have the right version, you can simply clone this repo and pip install

.. code-block:: bash

   git clone https://github.com/facebookresearch/ReAgent.git
   cd ReAgent
   pip install ".[gym]"

   # install nightly torch (change cpu to cu102 if fit)
   pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

If you don't want need gym dependencies, you can remove :code:`[gym]`

To verify your setup please run `tox <https://tox.readthedocs.io/en/latest/>`_.

.. code-block:: bash

   pip install tox
   tox

Spark preprocessing JAR
^^^^^^^^^^^^^^^^^^^^^^^

If you don't want to rebuild the JAR, you can grab `the pre-built version from CircleCI <https://841-98565575-gh.circle-artifacts.com/0/rl-preprocessing-1.1.jar>`_,
under `the artifact section of end_to_end_test <https://app.circleci.com/pipelines/github/facebookresearch/ReAgent?branch=master>`_.

To build from source, you'll need JDK, Scala, & Maven. We will use `SDKMAN! <https://sdkman.io/>`_ to install them.

.. code-block:: bash

   curl -s "https://get.sdkman.io" | bash
   source "$HOME/.sdkman/bin/sdkman-init.sh"
   sdk version
   sdk install java 8.0.292.hs-adpt
   sdk install scala
   sdk install maven

If you are testing locally, you can also install Spark

.. code-block:: bash

   sdk install spark 3.1.1

Now, you can build our preprocessing JAR

.. code-block:: bash

   mvn -f preprocessing/pom.xml clean package

RASP (Not Actively Maintained)
^^^^

RASP (ReAgent Serving Platform) is a decision-serving library. It also has standlone binary. It depends on libtorch,
which cannot be statically linked at the moment. Therefore, we don't have a pre-built version.

To build the CLI, you'll need `CMake <https://cmake.org/>`_ and the following libraries:

-  Nightly build of `libtorch <https://pytorch.org/cppdocs/>`_
- `boost <https://www.boost.org/>`_
- `gflags <https://gflags.github.io/gflags/>`_
- `glog <https://github.com/google/glog>`_
- `eigen <http://eigen.tuxfamily.org/>`_

If you don't have those requirements, one easy way to get them is through `conda`.
We recommend `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ for this purpose.
If you want to install those requirements via conda, you can type this in the shell:

.. code-block::

   conda install --file rasp_requirements.txt

To get libtorch, please go to `pytorch <https://pytorch.org/get-started/locally/>`_.
Please make sure to download the "Preview (Nightly)" since our code is tested with that version.
Below, we assumed you put the extracted file at :code:`$HOME/libtorch`.

You will also need to make sure to init git submodules

.. code-block::

   git submodule update --force --recursive --init --remote

Now, you are ready to build

.. code-block::

   mkdir -p serving/build
   cd serving/build
   cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_CXX_STANDARD=17 ..
   make
