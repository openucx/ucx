Operating System Limits
=======================


UCX can be affected by a variety of limits, not just defined by UCX itself but also by the operating system. In this section we describe some of the limits that may be encountered by the user when running UCXX or just UCX alone.

File Descriptors
----------------

In sockets-based connections, multiple file descriptors may be open to establish connections between endpoints. When UCX is establishing connection between endpoints via protocols such as TCP, an error such as below may occur:

::

    ucxx.exceptions.UCXError: User-defined limit was reached

One possible cause for this is that the limit established by the OS or system administrators has been reached by the user. This limit can be checked with:

::

    $ ulimit -n

If the user has permission to do so, the file descriptor limit can be increased by typing the new limit after the command above. For example, to set a new limit of 1 million, the following should be executed:

::

    $ ulimit -n 1000000

Another way the number of open files limit can be increased is by editing the limits.conf file in the operating system. Please consult your system administration for details.

Please note that the number of open files required may different according to the application, further investigation may be required to find optimal values.

For systems with specialized hardware such as InfiniBand, using RDMACM may also help circumventing that issue, as it doesn't rely heavily on file descriptors.


Maximum Connections
-------------------

UCX respects the operating system's limit of socket listen() backlog, known in userspace as SOMAXCONN. This limit may cause creating may cause new endpoints from connecting to a listener to hang if too many connections happen to be initiated too quickly.

To check for the current limit, the user can execute the following command:

::

    $ sysctl net.core.somaxconn

For most Linux distros, the default limit is 128. To increase that limit to 65535 for example, the user may run the following (require root or sudo permissions):

::

    $ sudo sysctl -w net.core.somaxconn=128
