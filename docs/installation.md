# Installation
---

##### Docker

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

Once the Docker image is built you can start an interactive shell in the container and run the unit tests. To have the ability to edit files locally and have changes be available in the Docker container, mount the local Horizon repo as a volume:
```
docker run -v /<LOCAL_PATH_TO_HORIZON>/Horizon:/home/Horizon -it horizon:dev
```

If you have SELinux (Fedora, Redhat, etc.) you will have to start docker with the following command (notice the `:Z` at the end of path):

```
docker run -v /<LOCAL_PATH_TO_HORIZON>/Horizon:/home/Horizon:Z -it horizon:dev
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
- OpenJDK 8
- Maven

To install them all, you can run `./install_compilers.sh`. After it finished, you will need to add this line to your `.bash_profile`

```
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
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
