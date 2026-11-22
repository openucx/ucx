# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
from random import random
from typing import List

from ucxx.examples.python_future_task_app import PythonFutureTaskApplication


def submit_task(
    cpp_app: PythonFutureTaskApplication,
    num_tasks: int = 10,
    max_task_duration: float = 1.0,
) -> List[asyncio.Future]:
    return [
        cpp_app.submit(duration=random() * max_task_duration, id=t)
        for t in range(num_tasks)
    ]


async def main():
    cpp_app = PythonFutureTaskApplication(asyncio.get_running_loop())
    num_tasks = 10
    max_task_duration = 3.0

    tasks = submit_task(
        cpp_app=cpp_app, num_tasks=num_tasks, max_task_duration=max_task_duration
    )
    print("Tasks submitted")
    results = await asyncio.gather(*tasks)
    print(f"Future {results=}", flush=True)
    assert all(got == expected for got, expected in zip(results, range(num_tasks)))


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
