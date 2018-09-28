# Docker image for Jenkins continuous integration test

`build.sh` is called by
https://github.com/pytorch/ossci-job-dsl/blob/master/src/jobs/horizon_docker.groovy
in order to build a Docker image with Horizon dependencies, for use
by Jenkins continuous integration tests.

`Dockerfile` contains the Docker image specifications, and calls
`install_prereqs.sh` and `add_jenkins_user.sh` as a part of installing the
dependencies. Note that the Docker image does NOT contain a copy of the
Horizon repo, as that is done by
https://github.com/pytorch/ossci-job-dsl/blob/master/src/jobs/horizon.groovy
(which in turn calls `../../.jenkins/build.sh` to actually build/install
Horizon and run tests).

## Building a new Docker image

Most changes to Horizon code will not require any change to the
Docker image used by Jenkins. However, you may need to build a new Docker image
if you:
1. Add a dependency to a new external package
2. Need to pull in an updated version of an external package (including
dependencies such as PyTorch, Caffe2, ONNX)

To do so, you need to push your changes to a branch of facebookresearch/Horizon.
If your pull request is from a local fork of that repo, you may need to
manually copy those changes to a new branch of facebookresearch/Horizon.
The following links may be useful:
* https://gist.github.com/IanVaughan/2887949
* https://help.github.com/articles/adding-a-remote/
* https://stackoverflow.com/questions/4878249/how-to-change-the-remote-a-branch-is-tracking

Afterwards, go to https://ci.pytorch.org/jenkins/job/horizon-docker-trigger/build
and build a new Docker image off that branch. Note the build # of your new
Docker image. You can check whether a particular Jenkins build/test is using
your Docker image by checking `DOCKER_IMAGE_TAG` under "Parameters" on the
side bar. You can re-run a Jenkins build/test using an updated Docker image by
clicking "Rebuild" on the side bar and updating `DOCKER_IMAGE_TAG` to the
build # of your new Docker image.

Assuming that tests at head all pass under the new Docker image,
horizon-docker-trigger should have automatically updated the version number in
https://github.com/pytorch/ossci-job-dsl/blob/master/src/main/groovy/ossci/horizon/DockerVersion.groovy.
However, if the Docker image change is tied to another pull request, and they're
both backward-incompatible, you may need to add some env-checking to the
pull request to make it compatible with both the old and new Docker images;
or temporarily break tests at head; or manually edit DockerVersion.groovy.

It may be possible to automate this if someone is interested in trying to
"fix the job dsl to automatically rebuild upon docker changes 🙂
(but it is not that easy; you need to only rebuild docker if there is
a change in the docker files; you don't want to keep rebuilding it for
non-Docker changes)" per @ezyang.
