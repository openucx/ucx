Quickstart
==========


Setup
-----

Create a new conda environment with UCXX:

::

    conda create -n ucxx -c conda-forge -c rapidsai \
        cuda-version=<CUDA version> ucxx

For a more detailed guide on installation options please refer to the :doc:`install` page.

Send/Recv NumPy Arrays
----------------------

Process 1 - Server
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucxx
    import numpy as np

    n_bytes = 2**30
    host = ucxx.get_address()  # if no suitable device is found, specify interface name with `ifname="..."`
    port = 13337


    async def send(ep):
        # recv buffer
        arr = np.empty(n_bytes, dtype="u1")
        await ep.recv(arr)
        assert np.count_nonzero(arr) == np.array(0, dtype=np.int64)
        print("Received NumPy array")

        # increment array and send back
        arr += 1
        print("Sending incremented NumPy array")
        await ep.send(arr)

        lf.close()


    async def main():
        global lf
        lf = ucxx.create_listener(send, port)

        while not lf.closed:
            await asyncio.sleep(0.1)

    if __name__ == "__main__":
        asyncio.run(main())


Process 2 - Client
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucxx
    import numpy as np

    port = 13337
    n_bytes = 2**30


    async def main():
        host = ucxx.get_address()  # if no suitable device is found, specify interface name with `ifname="..."`
        ep = await ucxx.create_endpoint(host, port)
        msg = np.zeros(n_bytes, dtype="u1")  # create some data to send

        # send message
        print("Send Original NumPy array")
        await ep.send(msg)  # send the real message

        # recv response
        print("Receive Incremented NumPy arrays")
        resp = np.empty_like(msg)
        await ep.recv(resp)  # receive the echo
        np.testing.assert_array_equal(msg + 1, resp)


    if __name__ == "__main__":
        asyncio.run(main())


Send/Recv CuPy Arrays
---------------------

.. note::
    If you are passing CuPy arrays between GPUs and want to use `NVLink <https://www.nvidia.com/en-us/data-center/nvlink/>`_ ensure you have correctly set ``UCX_TLS`` with ``cuda_ipc``. See the :doc:`configuration` for more details

Process 1 - Server
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucxx
    import cupy as cp

    n_bytes = 2**30
    host = ucxx.get_address()  # if no suitable device is found, specify interface name with `ifname="..."`
    port = 13337


    async def send(ep):
        # recv buffer
        arr = cp.empty(n_bytes, dtype="u1")
        await ep.recv(arr)
        assert cp.count_nonzero(arr) == cp.array(0, dtype=cp.int64)
        print("Received CuPy array")

        # increment array and send back
        arr += 1
        print("Sending incremented CuPy array")
        await ep.send(arr)

        lf.close()


    async def main():
        global lf
        lf = ucxx.create_listener(send, port)

        while not lf.closed:
            await asyncio.sleep(0.1)


    if __name__ == "__main__":
        asyncio.run(main())


Process 2 - Client
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import ucxx
    import cupy as cp

    port = 13337
    n_bytes = 2**30


    async def main():
        host = ucxx.get_address()  # if no suitable device is found, specify interface name with `ifname="..."`
        ep = await ucxx.create_endpoint(host, port)
        msg = cp.zeros(n_bytes, dtype="u1")  # create some data to send

        # send message
        print("Send Original CuPy array")
        await ep.send(msg)  # send the real message

        # recv response
        print("Receive Incremented CuPy arrays")
        resp = cp.empty_like(msg)
        await ep.recv(resp)  # receive the echo
        cp.testing.assert_array_equal(msg + 1, resp)

    if __name__ == "__main__":
        asyncio.run(main())
