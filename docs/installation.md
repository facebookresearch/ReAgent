# Installation

### Docker (recommended)

We have included a Dockerfile for the CPU-only build and CUDA build under the docker directory.
The CUDA build will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run.

To build, clone repository and cd into the respective directory:
```
git clone https://github.com/facebookresearch/Horizon.git
cd Horizon/docker/cpu/
```

On macOS you will need to increase the default memory allocation as the default of 2G is not enough. You can do this by clicking the whale icon in the task bar. We recommend using at least 8G of memory.

On macOS, you can then build the image:
```
docker build -t horizon:dev .
```
On Linux you can build the image with specific memory allocations from command line:
```
docker build -t horizon:dev --memory=8g --memory-swap=8g .
```

Once the Docker image is built you can start an interactive shell in the container and run the unit tests. To have the ability to edit files locally and have changes be available in the Docker container, mount the local Horizon repo as a volume using the `-v` flag. We also add `-p` for port mapping so we can view Tensorboard visualizations locally.
```
docker run -v /<LOCAL_PATH_TO_HORIZON>/Horizon:/home/Horizon -p 0.0.0.0:6006:6006 -it horizon:dev
```

To run with GPU, include `--runtime=nvidia` after installing [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

```
docker run --runtime=nvidia -v /<LOCAL_PATH_TO_HORIZON>/Horizon:/home/Horizon -p 0.0.0.0:6006:6006 -it horizon:dev
```

If you have SELinux (Fedora, Redhat, etc.) you will have to start docker with the following command (notice the `:Z` at the end of path):

```
docker run -v /<LOCAL_PATH_TO_HORIZON>/Horizon:/home/Horizon:Z -p 0.0.0.0:6006:6006 -it horizon:dev
```

To run with GPU, include `--runtime=nvidia` after installing [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

```
docker run --runtime=nvidia -v /<LOCAL_PATH_TO_HORIZON>/Horizon:/home/Horizon:Z -p 0.0.0.0:6006:6006 -it horizon:dev
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

### Anaconda

First, install anaconda from here (make sure to pick the python 3 version): [Website](https://www.anaconda.com/)

Switch to python 3.6 (3.7 has issues with tensorboard):
```
conda install python=3.6
```

Install OpenJDK and Maven:
```
conda install openjdk maven
```

Set JAVA_HOME to the location of your anaconda install
```
export JAVA_HOME=${HOME}/anaconda3
```

Install Spark (this installs to /usr/local/spark, but other directories are fine):
```
wget http://www-eu.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
tar -xzf spark-2.3.1-bin-hadoop2.7.tgz
mv spark-2.3.1-bin-hadoop2.7 /usr/local/spark
export PATH=$PATH:/usr/local/spark/bin
```

Clone Horizon repo:
```
git clone https://github.com/facebookresearch/Horizon.git
cd Horizon/
```

Install Horizon dependencies:
```
pip install -r requirements.txt
```

Then, install appropriate PyTorch 1.0 nightly build:
```
# For CPU build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# For CUDA 9.0 build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# For CUDA 9.2 build
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
```

And now, you are ready to install Horizon itself.

```
pip install -e .
```

At this point, you should be able to run all unit tests:

```
python setup.py test
```
