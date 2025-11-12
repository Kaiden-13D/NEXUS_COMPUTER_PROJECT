"""
Progress logging utility
"""
import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple fallback if tqdm is not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = kwargs.get('iterable', args[0] if args else [])
            self.total = kwargs.get('total', len(self.iterable) if hasattr(self.iterable, '__len__') else None)
            self.desc = kwargs.get('desc', '')
            self.n = 0
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.desc:
                print(f"{self.desc}: Complete")
        
        def __iter__(self):
            return iter(self.iterable)
        
        def update(self, n=1):
            self.n += n
            if self.total:
                pct = (self.n / self.total) * 100
                print(f"{self.desc}: {self.n}/{self.total} ({pct:.1f}%)", end='\r')
            else:
                print(f"{self.desc}: {self.n} items", end='\r')
        
        def set_description(self, desc):
            self.desc = desc


class ProgressLogger:
    """Progress logging class"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs/progress", timestamp_dir: str = "logs/timestamp"):
        """
        Args:
            experiment_name: Experiment name (bl1, bl2, bl3, abl1)
            log_dir: Progress log directory (not separated by experiment type)
            timestamp_dir: Timestamp log directory
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp_dir = Path(timestamp_dir)
        self.timestamp_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress log file (keep only latest - latest.log)
        self.log_file = self.log_dir / "latest.log"
        # Delete existing file and create new one
        if self.log_file.exists():
            self.log_file.unlink()
        self.log_file.touch()
        
        # Timestamp log file (keep only latest)
        self.timestamp_file = self.timestamp_dir / "latest.log"
        # Delete existing file and create new one
        if self.timestamp_file.exists():
            self.timestamp_file.unlink()
        self.timestamp_file.touch()
        
        print(f"Progress log: {self.log_file}", flush=True)
        print(f"Timestamp log: {self.timestamp_file}", flush=True)
    
    def log(self, message: str, print_to_console: bool = True, include_timestamp: bool = True):
        """
        Log message
        
        Args:
            message: Log message
            print_to_console: Whether to print to console
            include_timestamp: Whether to log to timestamp log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Write to progress log
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message)
        
        # Also write to timestamp log (keep only latest)
        if include_timestamp:
            with open(self.timestamp_file, 'a', encoding='utf-8') as f:
                f.write(log_message)
        
        if print_to_console:
            print(message)
    
    def tqdm(self, iterable, desc: str = "", total: Optional[int] = None, **kwargs):
        """
        tqdm wrapper (also logs to file)
        
        Args:
            iterable: Iterable object
            desc: Description
            total: Total count
            **kwargs: Additional tqdm options
        """
        if not TQDM_AVAILABLE:
            self.log(f"Starting: {desc}")
            return tqdm(iterable, desc=desc, total=total, **kwargs)
        
        # Configure tqdm
        pbar = tqdm(
            iterable,
            desc=desc,
            total=total,
            file=sys.stdout,
            **kwargs
        )
        
        # Initial log
        if desc:
            self.log(f"Starting: {desc}")
        
        # Periodically log progress
        original_update = pbar.update
        
        def update_with_log(n=1):
            result = original_update(n)
            if pbar.n % max(1, (pbar.total or 100) // 10) == 0 or pbar.n == (pbar.total or 0):
                pct = (pbar.n / pbar.total * 100) if pbar.total else 0
                self.log(f"{desc}: {pbar.n}/{pbar.total} ({pct:.1f}%)", print_to_console=False)
            return result
        
        pbar.update = update_with_log
        
        return pbar
    
    def log_complete(self, task_name: str):
        """Log task completion"""
        self.log(f"✓ Completed: {task_name}")


# Global instance (created per experiment)
_progress_logger: Optional[ProgressLogger] = None


def init_progress_logger(experiment_name: str):
    """Initialize progress logger"""
    global _progress_logger
    _progress_logger = ProgressLogger(experiment_name)
    return _progress_logger


def get_progress_logger() -> Optional[ProgressLogger]:
    """Get progress logger"""
    return _progress_logger

