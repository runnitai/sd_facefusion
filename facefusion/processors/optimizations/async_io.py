"""
Asynchronous I/O utilities for overlapping disk operations with GPU compute.
Provides producer/consumer patterns for frame reading/writing with configurable concurrency.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Queue, Empty, Full
from typing import List, Optional, Any, Dict

from facefusion import logger
from facefusion.vision import read_image, write_image

# CUDA thread safety flag
_FORCE_CPU_IN_WORKERS = True

@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame_number: int
    frame_path: str
    frame_data: Any = None
    metadata: Dict[str, Any] = None
    timestamp: float = 0.0


class AsyncFrameReader:
    """Asynchronous frame reader with configurable concurrency."""

    def __init__(self,
                 max_queue_size: int = 16,
                 num_workers: int = 4,
                 use_processes: bool = False):
        """
        Initialize async frame reader.
        
        Args:
            max_queue_size: Maximum number of frames to buffer
            num_workers: Number of worker threads/processes
            use_processes: Whether to use processes instead of threads
        """
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.use_processes = use_processes

        self.input_queue: Queue[str] = Queue(maxsize=max_queue_size)
        self.output_queue: Queue[FrameData] = Queue(maxsize=max_queue_size)
        self.executor = None
        self.futures = []
        self.is_running = False
        self.stats = {
            'frames_read': 0,
            'read_errors': 0,
            'start_time': 0,
            'total_read_time': 0
        }

        logger.debug(f"Initialized AsyncFrameReader: queue_size={max_queue_size}, "
                     f"workers={num_workers}, processes={use_processes}", __name__)

    def start(self) -> None:
        """Start the async frame reader."""
        if self.is_running:
            return

        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        self.executor = executor_class(max_workers=self.num_workers)
        self.is_running = True
        self.stats['start_time'] = time.time()

        logger.debug("AsyncFrameReader started", __name__)

    def stop(self) -> None:
        """Stop the async frame reader and cleanup resources."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel pending futures
        for future in self.futures:
            future.cancel()

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        # Clear queues
        self._clear_queue(self.input_queue)
        self._clear_queue(self.output_queue)

        self.futures.clear()

        logger.debug("AsyncFrameReader stopped", __name__)

    def _clear_queue(self, queue: Queue) -> None:
        """Clear all items from a queue."""
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break

    def _read_frame_worker(self, frame_path: str, frame_number: int) -> FrameData:
        """Worker function to read a single frame."""
        read_start = time.time()
        
        # Store original execution providers and force CPU for worker threads
        original_providers = None
        original_device_id = None
        if _FORCE_CPU_IN_WORKERS:
            try:
                from facefusion import state_manager
                from facefusion.inference_manager import clear_inference_pool
                
                # Store original settings
                original_providers = state_manager.get_item('execution_providers')
                original_device_id = state_manager.get_item('execution_device_id')
                
                if original_providers and 'cuda' in str(original_providers).lower():
                    # Temporarily force CPU execution in worker threads
                    state_manager.set_item('execution_providers', ['cpu'])
                    state_manager.set_item('execution_device_id', 'cpu')
                    
                    # Clear any existing CUDA inference pools to prevent context conflicts
                    try:
                        clear_inference_pool('face_detector')
                        clear_inference_pool('face_landmarker')
                        clear_inference_pool('face_recognizer')
                        clear_inference_pool('face_classifier')
                    except:
                        pass
                    
                    #logger.debug(f"Worker thread {threading.current_thread().name}: Forced CPU execution", __name__)
            except Exception as e:
                logger.debug(f"Could not override execution providers in worker: {e}", __name__)
        
        try:
            frame_data = read_image(frame_path)
            read_time = time.time() - read_start

            return FrameData(
                frame_number=frame_number,
                frame_path=frame_path,
                frame_data=frame_data,
                metadata={'read_time': read_time},
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Failed to read frame {frame_path}: {e}", __name__)
            self.stats['read_errors'] += 1
            return FrameData(
                frame_number=frame_number,
                frame_path=frame_path,
                frame_data=None,
                metadata={'error': str(e)},
                timestamp=time.time()
            )
        finally:
            # Restore original execution providers
            if _FORCE_CPU_IN_WORKERS and original_providers is not None:
                try:
                    state_manager.set_item('execution_providers', original_providers)
                    if original_device_id is not None:
                        state_manager.set_item('execution_device_id', original_device_id)
                except Exception as e:
                    logger.debug(f"Could not restore execution providers in worker: {e}", __name__)

    def add_frame_paths(self, frame_paths: List[str]) -> None:
        """Add frame paths to the reading queue."""
        if not self.is_running:
            self.start()

        # Submit reading tasks
        for frame_number, frame_path in enumerate(frame_paths):
            if not self.is_running:
                break

            future = self.executor.submit(self._read_frame_worker, frame_path, frame_number)
            self.futures.append(future)

        # Start monitoring completed futures
        self._monitor_futures()

    def _monitor_futures(self) -> None:
        """Monitor completed futures and add results to output queue."""

        def monitor():
            from facefusion import process_manager
            
            for future in as_completed(self.futures):
                # Check if processing should stop
                if not self.is_running or process_manager.is_stopping():
                    logger.debug("Stopping async frame reader due to process manager state", __name__)
                    break

                try:
                    frame_data = future.result()
                    if frame_data.frame_data is not None:
                        self.stats['frames_read'] += 1
                        if frame_data.metadata and 'read_time' in frame_data.metadata:
                            self.stats['total_read_time'] += frame_data.metadata['read_time']

                    # Add to output queue (block if queue is full)
                    while self.is_running and not process_manager.is_stopping():
                        try:
                            self.output_queue.put(frame_data, timeout=0.1)
                            break
                        except Full:
                            continue

                except Exception as e:
                    logger.error(f"Error processing future result: {e}", __name__)
                    self.stats['read_errors'] += 1

        # Run monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def get_frame(self, timeout: Optional[float] = None) -> Optional[FrameData]:
        """Get the next available frame."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get reading statistics."""
        elapsed_time = time.time() - self.stats['start_time'] if self.is_running else 0
        avg_read_time = (self.stats['total_read_time'] / self.stats['frames_read']
                         if self.stats['frames_read'] > 0 else 0)

        return {
            'frames_read': self.stats['frames_read'],
            'read_errors': self.stats['read_errors'],
            'elapsed_time': elapsed_time,
            'avg_read_time_ms': avg_read_time * 1000,
            'queue_size': self.output_queue.qsize(),
            'is_running': self.is_running
        }


class AsyncFrameWriter:
    """Asynchronous frame writer with configurable concurrency."""

    def __init__(self,
                 max_queue_size: int = 16,
                 num_workers: int = 2,
                 use_processes: bool = False):
        """
        Initialize async frame writer.
        
        Args:
            max_queue_size: Maximum number of frames to buffer for writing
            num_workers: Number of writer threads/processes
            use_processes: Whether to use processes instead of threads
        """
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.use_processes = use_processes

        self.input_queue: Queue[FrameData] = Queue(maxsize=max_queue_size)
        self.executor = None
        self.futures = []
        self.is_running = False
        self.stats = {
            'frames_written': 0,
            'write_errors': 0,
            'start_time': 0,
            'total_write_time': 0
        }

        logger.debug(f"Initialized AsyncFrameWriter: queue_size={max_queue_size}, "
                     f"workers={num_workers}, processes={use_processes}", __name__)

    def start(self) -> None:
        """Start the async frame writer."""
        if self.is_running:
            return

        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        self.executor = executor_class(max_workers=self.num_workers)
        self.is_running = True
        self.stats['start_time'] = time.time()

        # Start writer worker
        self._start_writer_worker()

        logger.debug("AsyncFrameWriter started", __name__)

    def stop(self) -> None:
        """Stop the async frame writer and cleanup resources."""
        if not self.is_running:
            return

        self.is_running = False

        # Wait for pending writes to complete
        for future in self.futures:
            try:
                future.result(timeout=5.0)  # Wait up to 5 seconds per write
            except:
                pass

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        # Clear queues
        self._clear_queue(self.input_queue)
        self.futures.clear()

        logger.debug("AsyncFrameWriter stopped", __name__)

    def _clear_queue(self, queue: Queue) -> None:
        """Clear all items from a queue."""
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break

    def _write_frame_worker(self, frame_data: FrameData, output_path: str) -> bool:
        """Worker function to write a single frame."""
        write_start = time.time()
        
        # Store original execution providers and force CPU for worker threads
        original_providers = None
        original_device_id = None
        if _FORCE_CPU_IN_WORKERS:
            try:
                from facefusion import state_manager
                from facefusion.inference_manager import clear_inference_pool
                
                # Store original settings
                original_providers = state_manager.get_item('execution_providers')
                original_device_id = state_manager.get_item('execution_device_id')
                
                if original_providers and 'cuda' in str(original_providers).lower():
                    # Temporarily force CPU execution in worker threads
                    state_manager.set_item('execution_providers', ['cpu'])
                    state_manager.set_item('execution_device_id', 'cpu')
                    # logger.debug(f"Writer worker thread {threading.current_thread().name}: Forced CPU execution", __name__)
            except Exception as e:
                logger.debug(f"Could not override execution providers in writer worker: {e}", __name__)
        
        try:
            if frame_data.frame_data is not None:
                success = write_image(output_path, frame_data.frame_data)
                write_time = time.time() - write_start

                if success:
                    self.stats['frames_written'] += 1
                    self.stats['total_write_time'] += write_time
                else:
                    self.stats['write_errors'] += 1

                return success
            else:
                self.stats['write_errors'] += 1
                return False

        except Exception as e:
            logger.error(f"Failed to write frame to {output_path}: {e}", __name__)
            self.stats['write_errors'] += 1
            return False
        finally:
            # Restore original execution providers
            if _FORCE_CPU_IN_WORKERS and original_providers is not None:
                try:
                    state_manager.set_item('execution_providers', original_providers)
                    if original_device_id is not None:
                        state_manager.set_item('execution_device_id', original_device_id)
                except Exception as e:
                    logger.debug(f"Could not restore execution providers in writer worker: {e}", __name__)

    def _start_writer_worker(self) -> None:
        """Start the writer worker thread."""
        
        def writer_worker():
            from facefusion import process_manager
            
            while self.is_running and not process_manager.is_stopping():
                try:
                    # Get frame data from input queue
                    frame_data = self.input_queue.get(timeout=1.0)
                    
                    # Process the frame
                    output_path = frame_data.metadata.get('output_path') if frame_data.metadata else None
                    if output_path:
                        success = self._write_frame_worker(frame_data, output_path)
                        if not success:
                            logger.warn(f"Failed to write frame {frame_data.frame_number}", __name__)
                    
                    self.input_queue.task_done()
                    
                except Empty:
                    # Timeout is normal, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error in writer worker: {e}", __name__)
                    self.stats['write_errors'] += 1
            
            logger.debug("Writer worker stopped", __name__)

        # Start writer worker thread
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        writer_thread.start()

    def add_frame(self, frame_data: FrameData, output_path: Optional[str] = None,
                  timeout: Optional[float] = None) -> bool:
        """Add a frame to the writing queue."""
        if not self.is_running:
            self.start()

        if output_path:
            if not frame_data.metadata:
                frame_data.metadata = {}
            frame_data.metadata['output_path'] = output_path

        try:
            self.input_queue.put(frame_data, timeout=timeout)
            return True
        except Full:
            logger.warn("Write queue is full, frame write may be delayed", __name__)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get writing statistics."""
        elapsed_time = time.time() - self.stats['start_time'] if self.is_running else 0
        avg_write_time = (self.stats['total_write_time'] / self.stats['frames_written']
                          if self.stats['frames_written'] > 0 else 0)

        return {
            'frames_written': self.stats['frames_written'],
            'write_errors': self.stats['write_errors'],
            'elapsed_time': elapsed_time,
            'avg_write_time_ms': avg_write_time * 1000,
            'queue_size': self.input_queue.qsize(),
            'is_running': self.is_running
        }


class AsyncIOManager:
    """Manager for coordinated async I/O operations."""

    def __init__(self,
                 read_queue_size: int = 16,
                 write_queue_size: int = 16,
                 read_workers: int = 4,
                 write_workers: int = 2):
        """Initialize async I/O manager."""
        self.reader = AsyncFrameReader(read_queue_size, read_workers)
        self.writer = AsyncFrameWriter(write_queue_size, write_workers)

    def start(self) -> None:
        """Start both reader and writer."""
        self.reader.start()
        self.writer.start()
        logger.debug("AsyncIOManager started", __name__)

    def stop(self) -> None:
        """Stop both reader and writer."""
        self.reader.stop()
        self.writer.stop()
        logger.debug("AsyncIOManager stopped", __name__)

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from reader and writer."""
        reader_stats = self.reader.get_stats()
        writer_stats = self.writer.get_stats()

        return {
            'reader': reader_stats,
            'writer': writer_stats,
            'io_efficiency': {
                'read_fps': reader_stats['frames_read'] / max(reader_stats['elapsed_time'], 0.001),
                'write_fps': writer_stats['frames_written'] / max(writer_stats['elapsed_time'], 0.001),
                'error_rate': ((reader_stats['read_errors'] + writer_stats['write_errors']) /
                               max(reader_stats['frames_read'] + writer_stats['frames_written'], 1))
            }
        }


# Global async I/O manager instance
_global_io_manager: Optional[AsyncIOManager] = None
_io_manager_lock = threading.Lock()


def get_global_io_manager() -> AsyncIOManager:
    """Get or create the global async I/O manager."""
    global _global_io_manager
    with _io_manager_lock:
        if _global_io_manager is None:
            # Configure based on system resources
            read_workers = min(8, (os.cpu_count() or 4) // 2)
            write_workers = min(4, (os.cpu_count() or 4) // 4)

            _global_io_manager = AsyncIOManager(
                read_queue_size=32,
                write_queue_size=16,
                read_workers=read_workers,
                write_workers=write_workers
            )
        return _global_io_manager


def cleanup_global_io_manager() -> None:
    """Cleanup the global async I/O manager."""
    global _global_io_manager
    with _io_manager_lock:
        if _global_io_manager:
            _global_io_manager.stop()
            _global_io_manager = None
