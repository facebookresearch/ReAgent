
##### Docker

We have included a Dockerfile for the CPU-only build and CUDA build under the docker directory.
The CUDA build will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run.

To build, clone repository and cd into the respective directory:
```
git clone https://github.com/facebookresearch/Horizon.git
cd Horizon/docker/cpu/
```

Run:
```
docker build -t horizon:dev --memory=8g --memory-swap=8g .
```

Once the Docker image is built you can start an interactive shell in the container and run the unit tests:
```
docker run -it horizon:dev
cd Horizon
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
- [Oracle Java 8](https://launchpad.net/~webupd8team/+archive/ubuntu/java)
- Maven

To install them all, you can run `./install_compilers.sh`. After it finished, you will need to add this line to your `.bash_profile`

```
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
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
