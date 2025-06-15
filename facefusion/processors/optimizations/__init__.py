"""
Processor optimizations package.
Provides GPU acceleration, async I/O, and batch scheduling optimizations.
"""

from facefusion import logger
from typing import Dict, Any

def configure_cuda_thread_safety(force_cpu_in_workers: bool = True) -> None:
    """Configure CUDA thread safety for optimization modules.
    
    Args:
        force_cpu_in_workers: If True, forces CPU execution in worker threads
                             to prevent CUDA context issues.
    """
    try:
        from . import async_io
        async_io._FORCE_CPU_IN_WORKERS = force_cpu_in_workers
        logger.info(f"CUDA thread safety configured: force_cpu_in_workers={force_cpu_in_workers}", __name__)
    except ImportError:
        logger.debug("async_io module not available for CUDA thread safety configuration", __name__)

def validate_optimization_compatibility() -> Dict[str, Any]:
    """Validate that optimizations are compatible with current system configuration."""
    compatibility = {
        'async_io': {'compatible': True, 'issues': []},
        'gpu_cv_ops': {'compatible': True, 'issues': []},
        'batch_scheduler': {'compatible': True, 'issues': []},
        'multi_gpu': {'compatible': True, 'issues': []}
    }
    
    try:
        from facefusion import state_manager
        from facefusion.execution import has_execution_provider
        
        # Check execution providers
        try:
            execution_providers = state_manager.get_item('execution_providers')
        except:
            execution_providers = []
            
        has_cuda = has_execution_provider('cuda')
        
        # Validate async_io compatibility
        if execution_providers and 'cuda' in str(execution_providers).lower() and has_cuda:
            try:
                from . import async_io
                if not getattr(async_io, '_FORCE_CPU_IN_WORKERS', True):
                    compatibility['async_io']['issues'].append(
                        'CUDA thread safety not enabled - may cause context errors'
                    )
            except ImportError:
                compatibility['async_io']['compatible'] = False
                compatibility['async_io']['issues'].append('async_io module not available')
        
        # Validate GPU CV ops
        try:
            from . import gpu_cv_ops
            if has_cuda and not gpu_cv_ops.check_gpu_availability():
                compatibility['gpu_cv_ops']['issues'].append(
                    'CUDA available but GPU CV operations disabled'
                )
        except ImportError:
            compatibility['gpu_cv_ops']['compatible'] = False
            compatibility['gpu_cv_ops']['issues'].append('gpu_cv_ops module not available')
        
        # Validate multi-GPU setup
        try:
            from facefusion.inference_manager import get_multi_gpu_stats
            multi_gpu_stats = get_multi_gpu_stats()
            gpu_count = len(multi_gpu_stats.get('memory_usage', {}))
            
            if gpu_count > 1:
                # Check if batch scheduler is configured for multi-GPU
                try:
                    from . import batch_scheduler
                    compatibility['multi_gpu']['gpu_count'] = gpu_count
                except ImportError:
                    compatibility['multi_gpu']['issues'].append(
                        'Multi-GPU detected but batch scheduler not available'
                    )
        except Exception as e:
            compatibility['multi_gpu']['issues'].append(f'Multi-GPU validation failed: {e}')
    
    except Exception as e:
        logger.error(f"Optimization compatibility check failed: {e}", __name__)
    
    return compatibility

def get_optimization_status() -> dict:
    """Get status of optimization modules."""
    status = {}
    
    try:
        from . import async_io
        status['async_io'] = {
            'available': True,
            'force_cpu_in_workers': getattr(async_io, '_FORCE_CPU_IN_WORKERS', True)
        }
    except ImportError:
        status['async_io'] = {'available': False}
    
    try:
        from . import gpu_cv_ops
        status['gpu_cv_ops'] = {
            'available': True,
            'cuda_available': gpu_cv_ops.check_gpu_availability()
        }
    except ImportError:
        status['gpu_cv_ops'] = {'available': False}
    
    try:
        from . import batch_scheduler
        status['batch_scheduler'] = {'available': True}
    except ImportError:
        status['batch_scheduler'] = {'available': False}
    
    # Add compatibility check
    status['compatibility'] = validate_optimization_compatibility()
    
    return status 