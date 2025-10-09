"""
System monitoring for GPU temperature, memory, and performance.
"""

import subprocess
import time
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SystemMetrics:
    """System metrics snapshot."""
    timestamp: float
    gpu_temp: float
    gpu_power: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    is_safe: bool


class SystemMonitor:
    """Monitor GPU and system metrics during experiments."""

    def __init__(self, max_temp: float = 85.0):
        """
        Initialize system monitor.

        Args:
            max_temp: Maximum safe GPU temperature
        """
        self.max_temp = max_temp
        self.metrics_history = []

    def get_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                gpu_temp = float(values[0]) if values[0] != '[Not Supported]' else 0.0
                gpu_power = float(values[1]) if values[1] != '[Not Supported]' else 0.0
                gpu_mem_used = float(values[2]) if values[2] != '[Not Supported]' else 0.0
                gpu_mem_total = float(values[3]) if values[3] != '[Not Supported]' else 0.0
                gpu_util = float(values[4]) if values[4] != '[Not Supported]' else 0.0

                metrics = SystemMetrics(
                    timestamp=time.time(),
                    gpu_temp=gpu_temp,
                    gpu_power=gpu_power,
                    gpu_memory_used=gpu_mem_used,
                    gpu_memory_total=gpu_mem_total,
                    gpu_utilization=gpu_util,
                    is_safe=(gpu_temp < self.max_temp)
                )

                self.metrics_history.append(metrics)
                return metrics

        except Exception as e:
            print(f"Warning: Could not read GPU metrics: {e}")
            return None

    def get_summary(self) -> Dict:
        """Get summary statistics from collected metrics."""
        if not self.metrics_history:
            return {}

        temps = [m.gpu_temp for m in self.metrics_history]
        memory = [m.gpu_memory_used for m in self.metrics_history]
        util = [m.gpu_utilization for m in self.metrics_history]

        return {
            'gpu_temp_max': max(temps),
            'gpu_temp_avg': sum(temps) / len(temps),
            'gpu_temp_min': min(temps),
            'gpu_memory_peak_mb': max(memory),
            'gpu_memory_avg_mb': sum(memory) / len(memory),
            'gpu_utilization_avg': sum(util) / len(util),
            'samples_collected': len(self.metrics_history)
        }

    def wait_for_cooling(self, target_temp: float = 75.0, timeout: int = 300):
        """
        Wait for GPU to cool down to target temperature.

        Args:
            target_temp: Target temperature in Celsius
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()
        print(f"Waiting for GPU to cool to {target_temp}°C...")

        while time.time() - start_time < timeout:
            metrics = self.get_metrics()
            if metrics and metrics.gpu_temp <= target_temp:
                print(f"GPU cooled to {metrics.gpu_temp:.1f}°C")
                return True

            if metrics:
                print(f"Current GPU temp: {metrics.gpu_temp:.1f}°C, waiting...")

            time.sleep(10)

        print(f"Warning: Cooling timeout after {timeout}s")
        return False
