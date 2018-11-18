# Installation

### Anaconda

First, install anaconda from here (make sure to pick the python 3 version): [Website](https://www.anaconda.com/).

Next, we're going to add some channels that we need for certain software:

```
conda config --add channels conda-forge # For ONNX/tensorboardX
conda config --add channels pytorch # For PyTorch
```

Clone and enter Horizon repo:
```
git clone https://github.com/facebookresearch/Horizon.git
cd Horizon/
```

Install dependencies:
```
conda install --file docker/requirements.txt
```

Install ONNX using pip, which builds the latest version from source:
```
pip install onnx
```

Set JAVA_HOME to the location of your anaconda install
```
export JAVA_HOME="$(dirname $(dirname -- `which conda`))"
```

Install Spark (the mv command may need to be done as root):
```
wget http://www-eu.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
tar -xzf spark-2.3.1-bin-hadoop2.7.tgz
mv spark-2.3.1-bin-hadoop2.7 /usr/local/spark
```

Add the spark bin directory to your path so your terminal can find `spark-submit`:
```
export PATH=$PATH:/usr/local/spark/bin
```

Install OpenAI Gym if you plan on following our [tutorial](usage.md):
```
pip install "gym[classic_control,box2d,atari]"
```

We use Apache Thrift to generate container classes and handle serialization to/from JSON.  To build our thrift classes:
```
thrift --gen py --out . ml/rl/thrift/core.thrift
```

And now, you are ready to install Horizon itself.  We use "-e" to create an ephemral package.  This means that you can make changes to Horizon and they will be reflected in the package immediately.

```
pip install -e .
```

At this point, you should be able to run all unit tests:

```
python setup.py test
```

### Docker

We have included a Dockerfile for the CPU-only build and CUDA build under the docker directory.
The CUDA build will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run.

To build, clone repository and cd into the respective directory:
```
git clone https://github.com/facebookresearch/Horizon.git
cd Horizon/docker/
```

On macOS you will need to increase the default memory allocation as the default of 2G is not enough. You can do this by clicking the whale icon in the task bar. We recommend using at least 8G of memory.

On macOS, you can then build the image:
```
docker build -f cpu.Dockerfile -t horizon:dev .
```
On Linux you can build the image with specific memory allocations from command line:
```
docker build -f cpu.Dockerfile -t horizon:dev --memory=8g --memory-swap=8g .
```

To build with CUDA support, use the corresponding dockerfile:

```
docker build -f cuda.Dockerfile -t horizon:dev .
```

Once the Docker image is built you can start an interactive shell in the container and run the unit tests. To have the ability to edit files locally and have changes be available in the Docker container, mount the local Horizon repo as a volume using the `-v` flag. We also add `-p` for port mapping so we can view Tensorboard visualizations locally.
```
docker run -v $PWD/../:/home/Horizon -p 0.0.0.0:6006:6006 -it horizon:dev
```

To run with GPU, include `--runtime=nvidia` after installing [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

```
docker run --runtime=nvidia -v $PWD/../:/home/Horizon -p 0.0.0.0:6006:6006 -it horizon:dev
```

If you have SELinux (Fedora, Redhat, etc.) you will have to start docker with the following command (notice the `:Z` at the end of path):

```
docker run -v $PWD/../:/home/Horizon:Z -p 0.0.0.0:6006:6006 -it horizon:dev
```

To run with GPU, include `--runtime=nvidia` after installing [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

```
docker run --runtime=nvidia -v $PWD/../:/home/Horizon:Z -p 0.0.0.0:6006:6006 -it horizon:dev
```

Depending on where your local Horizon copy is, you may need to white list your shared path via Docker -> Preferences... -> File Sharing.

Once inside the container, run the setup file:
```
cd Horizon
./scripts/setup.sh
```

Now you can run the tests:
```
python setup.py test
```
