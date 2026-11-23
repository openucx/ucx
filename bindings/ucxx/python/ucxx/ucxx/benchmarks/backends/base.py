# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from argparse import Namespace
from queue import Queue


class BaseServer(ABC):
    @abstractmethod
    def __init__(self, args: Namespace, queue: Queue):
        """
        Benchmark server.

        Parameters
        ----------
        args: argparse.Namespace
            Parsed command-line arguments that will be used as parameters during
            the `run` method.
        queue: Queue
            Queue object where server will put the port it is listening at.
        """
        pass

    @property
    @abstractmethod
    def has_cuda_support() -> bool:
        """
        Check whether server implementation supports CUDA memory transfers.

        Returns
        -------
        ret: bool
            `True` if CUDA is supported, `False` otherwise.
        """
        return False

    @abstractmethod
    def run(self):
        """
        Run the benchmark server.

        The server is executed as follows:
        1. Start the listener and put port where it is listening into the queue
           registered in constructor;
        2. Setup any additional context (Active Message registration, memory buffers
           to reuse, etc.);
        3. Transfer data back-and-forth with client;
        4. Shutdown server.
        """
        pass


class BaseClient(ABC):
    @abstractmethod
    def __init__(self, args: Namespace, queue: Queue, server_address: str, port: int):
        """
        Benchmark client.

        Parameters
        ----------
        args
            Parsed command-line arguments that will be used as parameters during
            the `run` method.
        queue
            Queue object where to put timing results.
        server_address
            Hostname or IP address where server is listening at.
        port
            Port where server is listening at.
        """
        pass

    @property
    @abstractmethod
    def has_cuda_support() -> bool:
        """
        Check whether client implementation supports CUDA memory transfers.

        Returns
        -------
        ret: bool
            `True` if CUDA is supported, `False` otherwise.
        """
        return False

    @abstractmethod
    def run(self):
        """
        Run the benchmark client.

        The client is executed as follows:
        1. Connects to listener;
        2. Setup any additional context (Active Message registration, memory buffers
           to reuse, etc.);
        3. Transfer data back-and-forth with server;
        4. Shutdown client;
        5. Put timing results into the queue registered in constructor.
        """
        pass

    def print_backend_specific_config(self):
        """
        Pretty print configuration specific to backend implementation.
        """
        pass
