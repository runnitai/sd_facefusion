from functools import lru_cache
from time import sleep
from typing import List, Dict, Optional
import threading
import psutil

import onnx
from onnxruntime import InferenceSession

from facefusion import process_manager, state_manager, logger
from facefusion.app_context import detect_app_context
from facefusion.execution import create_execution_providers, has_execution_provider
from facefusion.thread_helper import thread_lock
from facefusion.typing import DownloadSet, ExecutionProviderKey, InferencePool, InferencePoolSet, ModelInitializer

INFERENCE_POOLS: InferencePoolSet = \
    {
        'cli': {},  # type:ignore[typeddict-item]
        'ui': {}  # type:ignore[typeddict-item]
    }

# Multi-GPU management
GPU_DEVICE_MAP: Dict[str, List[str]] = {}  # model_context -> [device_ids]
GPU_LOAD_BALANCER: Dict[str, int] = {}  # model_context -> current_device_index
GPU_STATS: Dict[str, Dict] = {}  # device_id -> {memory_used, inference_count}
MULTI_GPU_LOCK = threading.Lock()


class MultiGPUInferenceWrapper:
    """Wrapper that manages multiple GPU instances of the same model for parallel batch processing."""
    
    def __init__(self, model_instances: Dict[str, InferenceSession], available_gpus: List[str]):
        self.model_instances = model_instances
        self.available_gpus = available_gpus
        self.current_gpu_index = 0
        self.lock = threading.Lock()
    
    def get_inputs(self):
        """Get model inputs from the first available instance."""
        first_instance = next(iter(self.model_instances.values()))
        return first_instance.get_inputs()
    
    def get_outputs(self):
        """Get model outputs from the first available instance.""" 
        first_instance = next(iter(self.model_instances.values()))
        return first_instance.get_outputs()
    
    def run(self, output_names, input_feed, run_options=None):
        """Single inference - use round-robin GPU selection."""
        with self.lock:
            # Round-robin GPU selection
            gpu_id = self.available_gpus[self.current_gpu_index]
            self.current_gpu_index = (self.current_gpu_index + 1) % len(self.available_gpus)
            
        # Run inference on selected GPU
        selected_instance = self.model_instances[gpu_id]
        return selected_instance.run(output_names, input_feed, run_options)
    
    def run_batch_parallel(self, output_names, input_feed_list, run_options=None):
        """Parallel batch inference across multiple GPUs."""
        import concurrent.futures
        
        if len(input_feed_list) == 1:
            # Single item, just use regular run
            return [self.run(output_names, input_feed_list[0], run_options)]
        
        # Distribute batch across available GPUs
        batch_size = len(input_feed_list)
        num_gpus = len(self.available_gpus)
        
        # Split batch into chunks for each GPU
        chunks = []
        chunk_size = max(1, batch_size // num_gpus)
        
        for i in range(0, batch_size, chunk_size):
            chunk = input_feed_list[i:i + chunk_size]
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
        
        # Process chunks in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_gpu = {}
            
            for i, chunk in enumerate(chunks):
                gpu_id = self.available_gpus[i % num_gpus]
                model_instance = self.model_instances[gpu_id]
                
                # Submit chunk processing to thread pool
                future = executor.submit(self._process_chunk, model_instance, output_names, chunk, run_options)
                future_to_gpu[future] = gpu_id
            
            # Collect results in order
            chunk_results = []
            for future in concurrent.futures.as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    chunk_result = future.result()
                    chunk_results.append(chunk_result)
                except Exception as e:
                    logger.error(f"GPU {gpu_id} processing failed: {e}", __name__)
                    # Fallback: process on main thread
                    chunk_result = self._process_chunk(
                        self.model_instances[self.available_gpus[0]], 
                        output_names, chunks[0], run_options
                    )
                    chunk_results.append(chunk_result)
        
        # Flatten results back to original order
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    def _process_chunk(self, model_instance: InferenceSession, output_names, input_chunk, run_options):
        """Process a chunk of inputs on a specific model instance."""
        chunk_results = []
        for input_feed in input_chunk:
            result = model_instance.run(output_names, input_feed, run_options)
            chunk_results.append(result)
        return chunk_results


def get_inference_pool(model_context: str, model_sources: DownloadSet,
                       preferred_provider: str = "default") -> InferencePool:
    global INFERENCE_POOLS

    with thread_lock():
        while process_manager.is_checking():
            sleep(0.5)
        app_context = detect_app_context()
        inference_context = get_inference_context(model_context, preferred_provider)
        requested_context = INFERENCE_POOLS.get(app_context).get(inference_context)
        if app_context == 'cli' and INFERENCE_POOLS.get('ui').get(inference_context) and not requested_context:
            INFERENCE_POOLS['cli'][inference_context] = INFERENCE_POOLS.get('ui').get(inference_context)
        if app_context == 'ui' and INFERENCE_POOLS.get('cli').get(inference_context) and not requested_context:
            INFERENCE_POOLS['ui'][inference_context] = INFERENCE_POOLS.get('cli').get(inference_context)
        if not requested_context:
            execution_provider_keys = resolve_execution_provider_keys(model_context, preferred_provider)
            print(f"Creating inference pool for {model_context} with {execution_provider_keys}.")
            
            # Check if multi-GPU optimization is enabled
            enable_multi_gpu = state_manager.get_item('enable_multi_gpu')
            use_multi_gpu = enable_multi_gpu and has_execution_provider('cuda')
            
            if use_multi_gpu:
                INFERENCE_POOLS[app_context][inference_context] = create_multi_gpu_inference_pool(
                    model_sources, execution_provider_keys)
                logger.info(f"Created multi-GPU inference pool for {model_context}", __name__)
            else:
                INFERENCE_POOLS[app_context][inference_context] = create_inference_pool(
                    model_sources, state_manager.get_item('execution_device_id'), execution_provider_keys)
                logger.info(f"Created single-GPU inference pool for {model_context}", __name__)

        return INFERENCE_POOLS.get(app_context).get(inference_context)


def create_inference_pool(model_sources: DownloadSet, execution_device_id: str,
                          execution_provider_keys: List[ExecutionProviderKey]) -> InferencePool:
    inference_pool: InferencePool = {}

    for model_name in model_sources.keys():
        inference_pool[model_name] = create_inference_session(model_sources.get(model_name).get('path'),
                                                              execution_device_id, execution_provider_keys)
    return inference_pool


def clear_inference_pool(model_context: str, preferred_provider: str = "default") -> None:
    global INFERENCE_POOLS

    app_context = detect_app_context()
    inference_context = get_inference_context(model_context, preferred_provider)

    if INFERENCE_POOLS.get(app_context).get(inference_context):
        del INFERENCE_POOLS[app_context][inference_context]


def create_inference_session(model_path: str, execution_device_id: str,
                             execution_provider_keys: List[ExecutionProviderKey]) -> InferenceSession:
    execution_providers = create_execution_providers(execution_device_id, execution_provider_keys)
    return InferenceSession(model_path, providers=execution_providers)


@lru_cache(maxsize=None)
def get_static_model_initializer(model_path: str) -> ModelInitializer:
    model = onnx.load(model_path)
    return onnx.numpy_helper.to_array(model.graph.initializer[-1])


def resolve_execution_provider_keys(model_context: str, preferred_provider: str = "default") -> List[
    ExecutionProviderKey]:
    if has_execution_provider('coreml') and ('age_modifier' in model_context or 'frame_colorizer' in model_context):
        return ['cpu']
    if preferred_provider != "default" and has_execution_provider(preferred_provider):
        print(f"Using preferred provider {preferred_provider} for {model_context}.")
        return [preferred_provider]

    return state_manager.get_item('execution_providers')


def get_inference_context(model_context: str, preferred_provider: str = "default") -> str:
    execution_provider_keys = resolve_execution_provider_keys(model_context, preferred_provider)
    inference_context = model_context + '.' + '_'.join(execution_provider_keys)
    return inference_context


def detect_available_gpus() -> List[str]:
    """Detect available GPU devices for inference."""
    available_gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # Only use GPUs with at least 4GB free memory
            if info.free > 4 * 1024 * 1024 * 1024:  # 4GB in bytes
                available_gpus.append(str(i))
                GPU_STATS[str(i)] = {
                    'memory_used': 0,
                    'inference_count': 0,
                    'total_memory': info.total,
                    'free_memory': info.free
                }
    except ImportError:
        logger.debug("pynvml not available, falling back to single GPU", __name__)
        available_gpus = ['0']  # Default to GPU 0
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}, using single GPU", __name__)
        available_gpus = ['0']
    
    return available_gpus


def create_multi_gpu_model_instances(model_name: str, model_path: str, available_gpus: List[str], execution_provider_keys: List[ExecutionProviderKey]) -> Dict[str, InferenceSession]:
    """Create multiple instances of the same model across available GPUs for parallel processing."""
    model_instances = {}
    
    for gpu_id in available_gpus:
        try:
            # Create inference session on specific GPU
            model_instances[gpu_id] = create_inference_session(model_path, gpu_id, execution_provider_keys)
            logger.info(f"Loaded {model_name} instance on GPU {gpu_id}", __name__)
        except Exception as e:
            logger.error(f"Failed to load {model_name} on GPU {gpu_id}: {e}", __name__)
    
    return model_instances


def get_optimal_device_for_model(model_context: str) -> str:
    """Get the optimal device for a model using load balancing."""
    with MULTI_GPU_LOCK:
        if model_context not in GPU_DEVICE_MAP:
            return state_manager.get_item('execution_device_id', '0')
        
        available_devices = GPU_DEVICE_MAP[model_context]
        if len(available_devices) == 1:
            return available_devices[0]
        
        # Round-robin load balancing
        current_index = GPU_LOAD_BALANCER.get(model_context, 0)
        selected_device = available_devices[current_index]
        
        # Update round-robin index
        GPU_LOAD_BALANCER[model_context] = (current_index + 1) % len(available_devices)
        
        # Update inference count
        if selected_device in GPU_STATS:
            GPU_STATS[selected_device]['inference_count'] += 1
        
        return selected_device


def create_multi_gpu_inference_pool(model_sources: DownloadSet, execution_provider_keys: List[ExecutionProviderKey]) -> InferencePool:
    """Create inference pool with multi-GPU support - loads same models on multiple GPUs for parallel processing."""
    inference_pool: InferencePool = {}
    
    # Detect available GPUs
    available_gpus = detect_available_gpus()
    logger.info(f"Detected {len(available_gpus)} available GPUs: {available_gpus}", __name__)
    
    if len(available_gpus) <= 1:
        # Fall back to single GPU behavior
        execution_device_id = state_manager.get_item('execution_device_id', '0')
        return create_inference_pool(model_sources, execution_device_id, execution_provider_keys)
    
    # For each model, create instances on all available GPUs
    for model_name in model_sources.keys():
        model_path = model_sources.get(model_name).get('path')
        
        # Create multi-GPU instances of this model
        model_instances = create_multi_gpu_model_instances(model_name, model_path, available_gpus, execution_provider_keys)
        
        if model_instances:
            # Store as a special multi-GPU inference pool structure
            inference_pool[model_name] = MultiGPUInferenceWrapper(model_instances, available_gpus)
            
            # Track GPU mapping for this model
            GPU_DEVICE_MAP[model_name] = available_gpus.copy()
            GPU_LOAD_BALANCER[model_name] = 0  # Start with GPU 0
            
            logger.info(f"Created multi-GPU pool for {model_name} across {len(model_instances)} GPUs", __name__)
        else:
            # Fallback to single GPU if multi-GPU failed
            default_device = state_manager.get_item('execution_device_id', '0')
            inference_pool[model_name] = create_inference_session(model_path, default_device, execution_provider_keys)
            logger.warning(f"Fell back to single GPU for {model_name}", __name__)
    
    return inference_pool


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage for monitoring."""
    usage = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usage[str(i)] = (info.used / info.total) * 100
    except:
        pass
    
    return usage


def get_multi_gpu_stats() -> Dict:
    """Get comprehensive multi-GPU statistics."""
    stats = {
        'device_map': GPU_DEVICE_MAP.copy(),
        'load_balancer': GPU_LOAD_BALANCER.copy(),
        'gpu_stats': GPU_STATS.copy(),
        'memory_usage': get_gpu_memory_usage()
    }
    return stats


def rebalance_gpu_load() -> None:
    """Rebalance GPU load based on current usage (advanced feature)."""
    memory_usage = get_gpu_memory_usage()
    
    # Find overloaded GPUs (>90% memory usage)
    overloaded_gpus = [gpu for gpu, usage in memory_usage.items() if usage > 90]
    
    if overloaded_gpus:
        logger.warning(f"Detected overloaded GPUs: {overloaded_gpus}", __name__)
        # Future enhancement: implement model migration between GPUs
        
    # Log current GPU distribution for monitoring
    logger.debug(f"Current GPU memory usage: {memory_usage}", __name__)
    logger.debug(f"Inference counts: {GPU_STATS}", __name__)
