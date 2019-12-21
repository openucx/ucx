.. 
.. Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
..
.. See file LICENSE for terms.
..

*******
OpenUCX
*******

Unified Communication X (UCX) provides an optimized communication layer for
Message Passing (MPI), PGAS/OpenSHMEM libraries and RPC/data-centric
applications.

UCX utilizes high-speed networks for inter-node communication, and shared
memory mechanisms for efficient intra-node communication.

.. image:: _static/UCX_Layers.png
   :alt: UCX layer diagram
   :align: center

.. toctree::
   :maxdepth: 3
   :hidden:

   download
   running
   faq


Quick start
***********

The following commands will download and build UCX v1.6 :doc:`release <download>`:

.. code-block:: console

    $ wget https://github.com/openucx/ucx/releases/download/v1.6.1/ucx-1.6.1.tar.gz
    $ tar xzf ucx-1.6.1.tar.gz
    $ cd ucx-1.6.1
    $ ./contrib/configure-release --prefix=$PWD/install
    $ make -j8 install


Documentation
*************

* TODO API
* `Examples <https://github.com/openucx/ucx/tree/v1.6.x/test/examples>`_


Projects using UCX
******************

* `OpenMPI <http://www.open-mpi.org>`_
* `MPICH <http://www.mpich.org>`_
* `OSSS shmem <http://github.com/openshmem-org/osss-ucx>`_
* `SparkUCX <http://github.com/openucx/sparkucx>`_


Developers section
******************

* `UCX on github <http://github.com/openucx/ucx>`_
* `Dev wiki <http://github.com/openucx/ucx/wiki>`_
* `Issue tracker <http://github.com/openucx/ucx/issues>`_
* `UCX mailing list <elist.ornl.gov/mailman/listinfo/ucx-group>`_


Buzz
****

* `UCX wins R&D 100 award <https://losalamosreporter.com/2019/11/07/nine-los-alamos-national-laboratory-projects-win-rd-100-awards>`_
* `UCX @ OpenSHMEM workshop <http://www.openucx.org/wp-content/uploads/2015/08/UCX_OpenSHMEM_2015.pdf>`_
