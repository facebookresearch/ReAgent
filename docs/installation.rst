.. _installation:

Installation
============

Anaconda
^^^^^^^^

First, install anaconda from here (make sure to pick the python 3 version): `Website <https://www.anaconda.com/>`_.  If you would like a smaller download, check out miniconda: `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Next, we're going to add some channels that we need for certain software:

.. code-block::

   conda config --add channels conda-forge # For ONNX/tensorboardX
   conda config --add channels pytorch # For PyTorch

If you get an error saying that the "conda" command could not be found, make sure that anaconda is installed and your path is set correctly.

Clone and enter ReAgent repo:

.. code-block::

   git clone --recurse-submodules https://github.com/facebookresearch/ReAgent.git
   cd ReAgent/

If you already cloned the repo without submodules, they can be added by running this command inside the repository"

.. code-block::

    git submodule update --init --recursive

Install dependencies:

.. code-block::

   conda install --file requirements.txt

Set JAVA_HOME to the location of your anaconda install

.. code-block::

   export JAVA_HOME="$(dirname $(dirname -- `which conda`))"

   echo $JAVA_HOME # Should see something like "/home/jjg/miniconda3"

Install Spark (the mv command may need to be done as root):

.. code-block::

   wget https://archive.apache.org/dist/spark/spark-2.3.3/spark-2.3.3-bin-hadoop2.7.tgz
   tar -xzf spark-2.3.3-bin-hadoop2.7.tgz
   sudo mv spark-2.3.3-bin-hadoop2.7 /usr/local/spark

Add the spark bin directory to your path so your terminal can find ``spark-submit``\ :

.. code-block::

   export PATH=$PATH:/usr/local/spark/bin

Install OpenAI Gym if you plan on following our `tutorial <usage.md>`_\ :

.. code-block::

   pip install "gym[classic_control,box2d,atari]"

Download libtorch from https://pytorch.org/get-started/locally/ and extract it to $HOME/libtorch

As of pytorch 1.3, libtorch is broken on OS/X.  To fix (mac only):

.. code-block::

    cp ~/miniconda3/lib/libiomp5.dylib $HOME/libtorch/lib/

And now, you are ready to install ReAgent itself.  To install the serving platform:

.. code-block::

    mkdir serving/build
    cd serving/build
    cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
    make


Next we must package the models.  We use "pip install -e" on the root directory of the repository to create an ephemral package.  This means that you can make changes to ReAgent and they will be reflected in the package immediately.

.. code-block::

   pip install -e .

At this point, you should be able to run all unit tests:

.. code-block::

   pytest
