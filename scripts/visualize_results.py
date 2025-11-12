#!/usr/bin/env python3
"""
Visualize experiment results from log files.

This script parses experiment log files and generates comprehensive visualizations
comparing different experiments (bl1, bl2, bl3, abl1, abl2).

Usage:
    python scripts/visualize_results.py [experiment_names...]
    
    If no experiment names are provided, it will visualize all available experiments.
    
Examples:
    python scripts/visualize_results.py bl1 bl2 bl3
    python scripts/visualize_results.py abl1
    python scripts/visualize_results.py  # Visualize all
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np


class ExperimentData:
    """Container for parsed experiment data"""
    def __init__(self, name: str):
        self.name = name
        self.rounds: List[int] = []
        self.global_acc: List[float] = []
        self.global_loss: List[float] = []
        self.personalized_acc: List[float] = []
        self.personalized_loss: List[float] = []
        self.round_cost_kb: List[float] = []
        self.round_time_s: List[float] = []
        self.cumulative_data_mb: List[float] = []
        self.cumulative_time_s: List[float] = []
        self.has_personalized = False


def parse_log_file(log_path: Path) -> Optional[ExperimentData]:
    """Parse a single log file and extract experiment data"""
    if not log_path.exists():
        return None
    
    # Determine experiment name from log file name
    exp_name = log_path.stem.split('_')[0]  # e.g., 'bl1_20251112_064959.log' -> 'bl1'
    data = ExperimentData(exp_name)
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern for round header: "=== Round X ==="
    round_pattern = r'=== Round (\d+) ==='
    rounds = re.findall(round_pattern, content)
    
    # Split content by rounds
    round_sections = re.split(round_pattern, content)
    
    # Skip first section (before first round)
    for i in range(1, len(round_sections), 2):
        if i + 1 >= len(round_sections):
            break
        
        round_num = int(round_sections[i])
        round_content = round_sections[i + 1]
        
        # Parse Global Accuracy and Loss
        perf_match = re.search(r'\[Perf\] Global GA: ([\d.]+)% \| Global Loss: ([\d.]+)', round_content)
        if perf_match:
            data.rounds.append(round_num)
            data.global_acc.append(float(perf_match.group(1)))
            data.global_loss.append(float(perf_match.group(2)))
        else:
            continue  # Skip this round if we can't parse it
        
        # Parse Personalized Accuracy and Loss (optional, for bl3, abl1, abl2)
        personalized_match = re.search(
            r'\[Perf\] Personalized PA: ([\d.]+)% \| Personalized Loss: ([\d.]+)',
            round_content
        )
        if personalized_match:
            data.has_personalized = True
            data.personalized_acc.append(float(personalized_match.group(1)))
            data.personalized_loss.append(float(personalized_match.group(2)))
        else:
            # Fill with None or previous value for consistency
            if data.has_personalized and len(data.personalized_acc) > 0:
                data.personalized_acc.append(data.personalized_acc[-1])
                data.personalized_loss.append(data.personalized_loss[-1])
            else:
                data.personalized_acc.append(None)
                data.personalized_loss.append(None)
        
        # Parse Round Cost and Time
        effi_match = re.search(
            r'\[Effi\] Round Cost: ([\d.]+) KB, \+([\d.]+)s simulated time',
            round_content
        )
        if effi_match:
            data.round_cost_kb.append(float(effi_match.group(1)))
            data.round_time_s.append(float(effi_match.group(2)))
        else:
            data.round_cost_kb.append(0.0)
            data.round_time_s.append(0.0)
        
        # Parse Cumulative Data and Time
        cumul_match = re.search(
            r'\[Cumul\] Total Data: ([\d.]+) MB \| Total Time: ([\d.]+)s',
            round_content
        )
        if cumul_match:
            data.cumulative_data_mb.append(float(cumul_match.group(1)))
            data.cumulative_time_s.append(float(cumul_match.group(2)))
        else:
            data.cumulative_data_mb.append(0.0)
            data.cumulative_time_s.append(0.0)
    
    return data if data.rounds else None


def find_latest_log(exp_name: str, logs_dir: Path = Path("logs")) -> Optional[Path]:
    """Find the latest log file for a given experiment"""
    exp_dir = logs_dir / exp_name
    if not exp_dir.exists():
        return None
    
    log_files = list(exp_dir.glob(f"{exp_name}_*.log"))
    if not log_files:
        return None
    
    # Return the most recently modified file
    return max(log_files, key=lambda p: p.stat().st_mtime)


def load_experiments(exp_names: List[str]) -> Dict[str, ExperimentData]:
    """Load experiment data from log files"""
    experiments = {}
    logs_dir = Path("logs")
    
    for exp_name in exp_names:
        log_path = find_latest_log(exp_name, logs_dir)
        if log_path:
            data = parse_log_file(log_path)
            if data:
                experiments[exp_name] = data
                print(f"Loaded {exp_name}: {len(data.rounds)} rounds from {log_path.name}")
            else:
                print(f"Warning: Could not parse {exp_name} log file")
        else:
            print(f"Warning: No log file found for {exp_name}")
    
    return experiments


def visualize_experiments(experiments: Dict[str, ExperimentData], output_path: Optional[Path] = None):
    """Create comprehensive visualizations for all experiments"""
    if not experiments:
        print("No experiment data to visualize")
        return
    
    # Color palette for different experiments
    colors = {
        'bl1': '#1f77b4',   # Blue
        'bl2': '#ff7f0e',   # Orange
        'bl3': '#2ca02c',   # Green
        'abl1': '#d62728',  # Red
        'abl2': '#9467bd',  # Purple
    }
    
    # Determine if any experiment has personalized metrics
    has_any_personalized = any(exp.has_personalized for exp in experiments.values())
    
    # Create figure with subplots
    if has_any_personalized:
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Global Accuracy (y-axis starts from 30%)
    ax1 = fig.add_subplot(gs[0, 0])
    for exp_name, data in experiments.items():
        color = colors.get(exp_name, '#000000')
        ax1.plot(data.rounds, data.global_acc, marker='o', label=exp_name.upper(), 
                color=color, linewidth=2, markersize=4, alpha=0.8)
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Global Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Global Accuracy vs Round', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim(bottom=30)
    
    # 2. Global Loss
    ax2 = fig.add_subplot(gs[0, 1])
    for exp_name, data in experiments.items():
        color = colors.get(exp_name, '#000000')
        ax2.plot(data.rounds, data.global_loss, marker='s', label=exp_name.upper(), 
                color=color, linewidth=2, markersize=4, alpha=0.8)
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Global Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Global Loss vs Round', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    # 3. Personalized Accuracy (if available)
    if has_any_personalized:
        ax3 = fig.add_subplot(gs[1, 0])
        for exp_name, data in experiments.items():
            if data.has_personalized:
                color = colors.get(exp_name, '#000000')
                # Filter out None values
                valid_rounds = [r for r, pa in zip(data.rounds, data.personalized_acc) if pa is not None]
                valid_acc = [pa for pa in data.personalized_acc if pa is not None]
                if valid_rounds:
                    ax3.plot(valid_rounds, valid_acc, marker='^', label=exp_name.upper(), 
                            color=color, linewidth=2, markersize=4, alpha=0.8)
        ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Personalized Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Personalized Accuracy vs Round', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        ax3.set_ylim(bottom=40)
        
        # 4. Personalized Loss (if available)
        ax4 = fig.add_subplot(gs[1, 1])
        for exp_name, data in experiments.items():
            if data.has_personalized:
                color = colors.get(exp_name, '#000000')
                valid_rounds = [r for r, pl in zip(data.rounds, data.personalized_loss) if pl is not None]
                valid_loss = [pl for pl in data.personalized_loss if pl is not None]
                if valid_rounds:
                    ax4.plot(valid_rounds, valid_loss, marker='v', label=exp_name.upper(), 
                            color=color, linewidth=2, markersize=4, alpha=0.8)
        ax4.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Personalized Loss', fontsize=12, fontweight='bold')
        ax4.set_title('Personalized Loss vs Round', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=10)
        
        # 5. Round Communication Cost (per round)
        ax5 = fig.add_subplot(gs[2, 0])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax5.plot(data.rounds, data.round_cost_kb, marker='o', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax5.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Round Communication Cost (KB)', fontsize=12, fontweight='bold')
        ax5.set_title('Round Communication Cost vs Round', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='best', fontsize=10)
        
        # 6. Round Simulated Time (per round)
        ax6 = fig.add_subplot(gs[2, 1])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax6.plot(data.rounds, data.round_time_s, marker='s', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax6.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Round Simulated Time (seconds)', fontsize=12, fontweight='bold')
        ax6.set_title('Round Simulated Time vs Round', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='best', fontsize=10)
        
        # 7. Cumulative Communication Cost
        ax7 = fig.add_subplot(gs[3, 0])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax7.plot(data.rounds, data.cumulative_data_mb, marker='o', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax7.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Cumulative Communication Cost (MB)', fontsize=12, fontweight='bold')
        ax7.set_title('Cumulative Communication Cost vs Round', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend(loc='best', fontsize=10)
        
        # 8. Cumulative Time
        ax8 = fig.add_subplot(gs[3, 1])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax8.plot(data.rounds, data.cumulative_time_s, marker='s', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax8.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Cumulative Time (seconds)', fontsize=12, fontweight='bold')
        ax8.set_title('Cumulative Time vs Round', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.legend(loc='best', fontsize=10)
    else:
        # Without personalized metrics, use 3x2 layout
        # 3. Round Communication Cost (per round)
        ax3 = fig.add_subplot(gs[1, 0])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax3.plot(data.rounds, data.round_cost_kb, marker='o', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Round Communication Cost (KB)', fontsize=12, fontweight='bold')
        ax3.set_title('Round Communication Cost vs Round', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        
        # 4. Round Simulated Time (per round)
        ax4 = fig.add_subplot(gs[1, 1])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax4.plot(data.rounds, data.round_time_s, marker='s', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax4.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Round Simulated Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('Round Simulated Time vs Round', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=10)
        
        # 5. Cumulative Communication Cost
        ax5 = fig.add_subplot(gs[2, 0])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax5.plot(data.rounds, data.cumulative_data_mb, marker='o', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax5.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Cumulative Communication Cost (MB)', fontsize=12, fontweight='bold')
        ax5.set_title('Cumulative Communication Cost vs Round', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='best', fontsize=10)
        
        # 6. Cumulative Time
        ax6 = fig.add_subplot(gs[2, 1])
        for exp_name, data in experiments.items():
            color = colors.get(exp_name, '#000000')
            ax6.plot(data.rounds, data.cumulative_time_s, marker='s', label=exp_name.upper(), 
                    color=color, linewidth=2, markersize=4, alpha=0.8)
        ax6.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Cumulative Time (seconds)', fontsize=12, fontweight='bold')
        ax6.set_title('Cumulative Time vs Round', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='best', fontsize=10)
    
    plt.suptitle('Experiment Results Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main entry point"""
    # Default experiments to visualize
    default_experiments = ['bl1', 'bl2', 'bl3', 'abl1', 'abl2']
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        exp_names = [name.lower() for name in sys.argv[1:]]
    else:
        exp_names = default_experiments
    
    print("Loading experiment data...")
    experiments = load_experiments(exp_names)
    
    if not experiments:
        print("No valid experiment data found. Please check log files in logs/ directory.")
        return
    
    print(f"\nVisualizing {len(experiments)} experiment(s): {', '.join(experiments.keys())}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    exp_str = "_".join(sorted(experiments.keys()))
    output_path = output_dir / f"comparison_{exp_str}.png"
    
    visualize_experiments(experiments, output_path)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

