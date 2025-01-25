import subprocess
import re
import os


def get_available_gpus(min_memory=100, excluded_gpus=[]):
    """
    Returns a list of available GPU ids that have more than 'min_memory' MiB of free memory.

    Args:
    min_memory (int): Minimum amount of free memory (in MiB) required for a GPU to be considered available.

    Returns:
    list: A list of GPU ids that meet the memory requirement.
    """
    try:
        # Run nvidia-smi to get memory usage
        smi_output = subprocess.check_output(
            'nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits',
            shell=True
        ).decode('utf-8')

        # Parse the output
        available_gpus = []
        for line in smi_output.strip().split('\n'):
            gpu_id, free_memory = line.split(',')
            if int(free_memory.strip()) >= min_memory:
                available_gpus.append(int(gpu_id.strip()))
        available_gpus = [gpu for gpu in available_gpus if gpu not in excluded_gpus]
        return available_gpus

    except subprocess.CalledProcessError as e:
        print(f"Failed to run nvidia-smi: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_gpu_memory(gpu_id):
    """Returns the free memory of the specified GPU."""
    try:
        # Run nvidia-smi command to check free memory for the specific GPU
        command = f'nvidia-smi -i {gpu_id} --query-gpu=memory.free --format=csv,nounits,noheader'
        memory_free = int(subprocess.check_output(command, shell=True))
        return memory_free
    except subprocess.CalledProcessError as e:
        print(f"Error checking memory for GPU {gpu_id}: {e}")
        return 0  # Assume no memory is free if there is an error


def set_gpu(gpu_id):
    """
    Set the GPU id for the process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
