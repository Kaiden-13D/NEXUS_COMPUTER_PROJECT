"""
진행 상황 로깅 유틸리티
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
    # tqdm이 없으면 간단한 fallback
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
    """진행 상황 로깅 클래스"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs/progress", timestamp_dir: str = "logs/timestamp"):
        """
        Args:
            experiment_name: 실험 이름 (bl1, bl2, bl3, abl1)
            log_dir: 진행 상황 로그 디렉토리 (실험 종류 구분 없음)
            timestamp_dir: 타임스탬프 로그 디렉토리
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp_dir = Path(timestamp_dir)
        self.timestamp_dir.mkdir(parents=True, exist_ok=True)
        
        # 진행 상황 로그 파일 (최신 1개만 유지 - latest.log)
        self.log_file = self.log_dir / "latest.log"
        # 기존 파일 삭제 후 새로 생성
        if self.log_file.exists():
            self.log_file.unlink()
        self.log_file.touch()
        
        # 타임스탬프 로그 파일 (최신 1개만 유지)
        self.timestamp_file = self.timestamp_dir / "latest.log"
        # 기존 파일 삭제 후 새로 생성
        if self.timestamp_file.exists():
            self.timestamp_file.unlink()
        self.timestamp_file.touch()
        
        print(f"Progress log: {self.log_file}", flush=True)
        print(f"Timestamp log: {self.timestamp_file}", flush=True)
    
    def log(self, message: str, print_to_console: bool = True, include_timestamp: bool = True):
        """
        로그 메시지 기록
        
        Args:
            message: 로그 메시지
            print_to_console: 콘솔에도 출력할지 여부
            include_timestamp: 타임스탬프 로그에도 기록할지 여부
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # 진행 상황 로그에 기록
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message)
        
        # 타임스탬프 로그에도 기록 (최신 1개만 유지)
        if include_timestamp:
            with open(self.timestamp_file, 'a', encoding='utf-8') as f:
                f.write(log_message)
        
        if print_to_console:
            print(message)
    
    def tqdm(self, iterable, desc: str = "", total: Optional[int] = None, **kwargs):
        """
        tqdm 래퍼 (로그 파일에도 기록)
        
        Args:
            iterable: 반복 가능한 객체
            desc: 설명
            total: 전체 개수
            **kwargs: tqdm 추가 옵션
        """
        if not TQDM_AVAILABLE:
            self.log(f"Starting: {desc}")
            return tqdm(iterable, desc=desc, total=total, **kwargs)
        
        # tqdm 설정
        pbar = tqdm(
            iterable,
            desc=desc,
            total=total,
            file=sys.stdout,
            **kwargs
        )
        
        # 초기 로그
        if desc:
            self.log(f"Starting: {desc}")
        
        # 진행 상황을 주기적으로 로그에 기록
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
        """작업 완료 로그"""
        self.log(f"✓ Completed: {task_name}")


# 전역 인스턴스 (실험별로 생성)
_progress_logger: Optional[ProgressLogger] = None


def init_progress_logger(experiment_name: str):
    """진행 상황 로거 초기화"""
    global _progress_logger
    _progress_logger = ProgressLogger(experiment_name)
    return _progress_logger


def get_progress_logger() -> Optional[ProgressLogger]:
    """진행 상황 로거 가져오기"""
    return _progress_logger

