import platform

import torch

from facefusion import globals

if platform.system().lower() == 'windows':
    import ctypes
else:
    import resource


def limit_system_memory(system_memory_limit: int = 1) -> bool:
    if platform.system().lower() == 'darwin':
        system_memory_limit = system_memory_limit * (1024 ** 6)
    else:
        system_memory_limit = system_memory_limit * (1024 ** 3)
    try:
        if platform.system().lower() == 'windows':
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(system_memory_limit), ctypes.c_size_t(
                system_memory_limit))  # type: ignore[attr-defined]
        else:
            resource.setrlimit(resource.RLIMIT_DATA, (system_memory_limit, system_memory_limit))
        return True
    except Exception:
        return False


def get_total_vram():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return total_memory / (1024 ** 2)  # Convert bytes to MB
    else:
        return 0


def tune_performance():
    queue_size = 1
    execution_thread_count = 1
    memory_strategy = "strict"
    vram = get_total_vram()
    if vram <= 8192:
        execution_thread_count = 6
    elif vram <= 16384:
        execution_thread_count = 10
        memory_strategy = "tolerant"
    elif vram <= 24576:
        execution_thread_count = 16
        queue_size = 2
        memory_strategy = "tolerant"
    elif vram <= 32768:
        execution_thread_count = 32
        queue_size = 2
        memory_strategy = "tolerant"
    globals.execution_thread_count = execution_thread_count
    globals.execution_queue_count = queue_size
    globals.memory_strategy = memory_strategy
    return execution_thread_count, queue_size, memory_strategy

