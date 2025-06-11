"""
Adaptive batch scheduler for dynamic batch size optimization.
Automatically adjusts batch sizes based on target latency and system performance.
"""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, List, Dict, Optional

from facefusion import logger


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""
    batch_size: int
    latency_ms: float
    throughput_fps: float
    memory_usage_mb: Optional[float] = None
    timestamp: float = 0.0


class BatchScheduler:
    """Adaptive batch scheduler that optimizes batch size based on performance metrics."""

    def __init__(self,
                 target_latency_ms: float = 100.0,
                 min_batch_size: int = 1,
                 max_batch_size: int = 8,
                 adjustment_factor: float = 0.1,
                 stability_threshold: int = 3):
        """
        Initialize batch scheduler.
        
        Args:
            target_latency_ms: Target latency in milliseconds
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            adjustment_factor: How aggressively to adjust batch size (0.0-1.0)
            stability_threshold: Number of stable measurements before adjusting
        """
        self.target_latency_ms = target_latency_ms
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adjustment_factor = adjustment_factor
        self.stability_threshold = stability_threshold

        self.current_batch_size = min_batch_size
        self.metrics_history: List[BatchMetrics] = []
        self.stable_count = 0
        self.lock = Lock()

        # Tolerance ranges
        self.latency_tolerance = 0.2  # 20% tolerance
        self.performance_window = 10  # Keep last N measurements

        logger.debug(f"Initialized BatchScheduler: target={target_latency_ms}ms, "
                     f"range=[{min_batch_size}, {max_batch_size}]", __name__)

    def record_batch_performance(self, batch_size: int, latency_ms: float,
                                 throughput_fps: float, memory_usage_mb: Optional[float] = None) -> None:
        """Record performance metrics for a batch."""
        with self.lock:
            metrics = BatchMetrics(
                batch_size=batch_size,
                latency_ms=latency_ms,
                throughput_fps=throughput_fps,
                memory_usage_mb=memory_usage_mb,
                timestamp=time.time()
            )

            self.metrics_history.append(metrics)

            # Keep only recent metrics
            if len(self.metrics_history) > self.performance_window:
                self.metrics_history.pop(0)

            # Update batch size based on performance
            self._update_batch_size(metrics)

    def _update_batch_size(self, latest_metrics: BatchMetrics) -> None:
        """Update batch size based on latest performance metrics."""
        current_latency = latest_metrics.latency_ms
        target_latency = self.target_latency_ms

        # Calculate performance ratio
        latency_ratio = current_latency / target_latency

        # Determine if we're in acceptable range
        lower_bound = 1.0 - self.latency_tolerance
        upper_bound = 1.0 + self.latency_tolerance

        if lower_bound <= latency_ratio <= upper_bound:
            # Performance is acceptable, increment stability counter
            self.stable_count += 1

            # If we've been stable for a while, try to increase batch size slightly
            if (self.stable_count >= self.stability_threshold and
                    self.current_batch_size < self.max_batch_size and
                    latency_ratio < 0.8):  # Only if we have headroom
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
                self.stable_count = 0
                logger.debug(f"Increased batch size to {self.current_batch_size} (stable performance)", __name__)
        else:
            # Performance is not acceptable, adjust batch size
            self.stable_count = 0

            if latency_ratio > upper_bound:
                # Too slow, decrease batch size
                new_batch_size = max(self.min_batch_size,
                                     int(self.current_batch_size * (1.0 - self.adjustment_factor)))
                if new_batch_size != self.current_batch_size:
                    self.current_batch_size = new_batch_size
                    logger.debug(f"Decreased batch size to {self.current_batch_size} "
                                 f"(latency {current_latency:.1f}ms > target {target_latency:.1f}ms)", __name__)

            elif latency_ratio < lower_bound:
                # Too fast, can potentially increase batch size
                if self.current_batch_size < self.max_batch_size:
                    new_batch_size = min(self.max_batch_size,
                                         int(self.current_batch_size * (1.0 + self.adjustment_factor)))
                    if new_batch_size != self.current_batch_size:
                        self.current_batch_size = new_batch_size
                        logger.debug(f"Increased batch size to {self.current_batch_size} "
                                     f"(latency {current_latency:.1f}ms < target {target_latency:.1f}ms)", __name__)

    def get_optimal_batch_size(self) -> int:
        """Get the current optimal batch size."""
        with self.lock:
            return self.current_batch_size

    def reset(self) -> None:
        """Reset scheduler state."""
        with self.lock:
            self.current_batch_size = self.min_batch_size
            self.metrics_history.clear()
            self.stable_count = 0
            logger.debug("BatchScheduler reset", __name__)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance metrics."""
        with self.lock:
            if not self.metrics_history:
                return {}

            recent_metrics = self.metrics_history[-5:]  # Last 5 measurements

            avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput_fps for m in recent_metrics) / len(recent_metrics)

            return {
                'current_batch_size': self.current_batch_size,
                'target_latency_ms': self.target_latency_ms,
                'avg_latency_ms': avg_latency,
                'avg_throughput_fps': avg_throughput,
                'measurements_count': len(self.metrics_history),
                'stable_count': self.stable_count
            }


# Global scheduler instances for different processors
_scheduler_instances: Dict[str, BatchScheduler] = {}
_scheduler_lock = Lock()


def get_scheduler_for_processor(processor_name: str, mode: str = "default") -> BatchScheduler:
    """Get or create a batch scheduler for a specific processor."""
    with _scheduler_lock:
        key = f"{processor_name}_{mode}"
        if key not in _scheduler_instances:
            if mode == "preview" or mode == "interactive":
                # Low latency for real-time processing
                scheduler = BatchScheduler(
                    target_latency_ms=33.0,  # ~30 FPS
                    min_batch_size=1,
                    max_batch_size=4,
                    adjustment_factor=0.15
                )
            elif mode == "offline" or mode == "batch":
                # High throughput for offline processing
                scheduler = BatchScheduler(
                    target_latency_ms=100.0,  # Higher latency acceptable
                    min_batch_size=1,
                    max_batch_size=16,
                    adjustment_factor=0.1
                )
            else:
                # Balanced default
                scheduler = BatchScheduler(
                    target_latency_ms=50.0,
                    min_batch_size=1,
                    max_batch_size=8,
                    adjustment_factor=0.12
                )

            _scheduler_instances[key] = scheduler
            logger.debug(f"Created BatchScheduler for {processor_name} in {mode} mode", __name__)
        return _scheduler_instances[key]


def clear_all_schedulers() -> None:
    """Clear all scheduler instances."""
    with _scheduler_lock:
        _scheduler_instances.clear()
        logger.debug("Cleared all BatchScheduler instances", __name__)
