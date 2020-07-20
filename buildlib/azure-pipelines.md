# Introduction

This project uses Azure Pipelines a GitHub check to validate pull requests
prior to merging. Each time a pull request is updated AZP will spawn VMs and
run compiles and tests based on the instructions in the
buildlib/azure-pipelines.yml file.

The test console output is linked from the GitHub check integration.

Azure Pipelines is linked to the UCF Consortium's Azure Tenant:

   https://portal.azure.com

And runs inside the Azure Dev Ops Organization:

   https://dev.azure.com/ucfconsort

As the UCX project:

   https://dev.azure.com/ucfconsort/ucx

# Containers

Most of the build steps are done inside Docker containers. The container
allows direct control and customization over the operating system environment
to achieve the required test.

UCF hosts a private docker registry on the Azure Container Registry at
ucfconsort.azurecr.io:

  https://portal.azure.com/#@jgunthorpegmail.onmicrosoft.com/resource/subscriptions/b8ff5e38-a317-4bbd-9831-b73d3887df30/resourceGroups/PipelinesRG/providers/Microsoft.ContainerRegistry/registries/ucfconsort/overview

The Azure Pipelines VM's have high speed access to this registry and can boot
containers failure quickly.

## Dockerfiles

Each container is described by a docker file in buildlib/. Dockerfiles can be
built locally using the build command at the top of the Dockerfile. Every
container has a unique name and tag reflecting its content. So that builds
continue to work on any stable branches the container version number should be
incremented when a build-incompatible change is made.

Once built the docker container needs to be pushed to the ACR, using the
following steps:

```shell
$ az login
$ az acr login --name ucfconsort
$ docker push ucfconsort.azurecr.io/ucx/centos7:1
```

See https://docs.microsoft.com/en-us/cli/azure for details on how to get the
command line tools.

## Alternate to 'docker push'

If network connectivity is too poor for push, then the container can be built
on a VM inside Azure using this command:

```shell
$ az acr build --registry ucfconsort -t ucfconsort.azurecr.io/ucx/centos7:1 -f buildlib/centos7.Dockerfile buildlib/
```

## Testing Containers Locally

The local container can be entered and checked out using a command sequence
similar to:

```shell
$ cd ../../ucx
$ docker run --rm -ti -v `pwd`:`pwd` -w `pwd` ucfconsort.azurecr.io/ucx/centos7:1 /bin/bash
# mkdir build-centos7 && cd build-centos7
# ../configure
# make
```

This will duplicate what will happen when running inside AZP.

# Release images
To build release images there is a `docker-compose` config. Here is how to use it:
```sh
cd buildlib
docker-compose build
```

Tag and push release images:
```sh
./buildlib/push-release-images.sh
```

