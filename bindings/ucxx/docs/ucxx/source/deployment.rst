NVLink and Docker/Kubernetes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use NVLink when running in containers using Docker and/or
Kubernetes the processes must share an IPC namespace for NVLink to work
correctly.

Many GPUs in one container
^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to ensure that processing accessing GPUs share an IPC
namespace is to run the processes within the same container. This means
exposing multiple GPUs to a single container.

Many containers with a shared IPC namespace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to isolate your processes into multiple containers and
expose one or more GPUs to each container you need to ensure they are
using a shared IPC namespace.

In a Docker configuration you can mark one container as having a
shareable IPC namespace with the flag ``--ipc="shareable"``. Other
containers can then share that namespace with the flag
``--ipc="container: <_name-or-ID_>"`` and passing the name or ID of the
container that is sharing it’s namespace.

You can also share the host IPC namespace with your container with the
flag ``--ipc="host"``, however this is not recommended on multi-tenant
hosts.

Privileged pods in a Kubernetes cluster `can also be configured to share
the host IPC`_.

For more information see the `Docker documentation`_.

.. _can also be configured to share the host IPC: https://kubernetes.io/docs/concepts/policy/pod-security-policy/#host-namespaces
.. _Docker documentation: https://docs.docker.com/engine/reference/run/#ipc-settings---ipc
