import os


def configure_runtime(gpu_id: str = "0") -> None:
    """Set GPU / JAX / PyTorch env vars before importing torch or JAX."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
