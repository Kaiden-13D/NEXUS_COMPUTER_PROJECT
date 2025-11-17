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
        
        # Parse Avg Test Accuracy and Loss (new unified format)
        perf_match = re.search(r'\[Perf\] Avg Test Acc: ([\d.]+)% \| Avg Loss: ([\d.]+)', round_content)
        if perf_match:
            data.rounds.append(round_num)
            data.global_acc.append(float(perf_match.group(1)))  # Store as global_acc for compatibility
            data.global_loss.append(float(perf_match.group(2)))  # Store as global_loss for compatibility
            # Also store as personalized (same value, unified metric)
            data.has_personalized = True
            data.personalized_acc.append(float(perf_match.group(1)))
            data.personalized_loss.append(float(perf_match.group(2)))
        else:
            # Try old format for backward compatibility
            old_perf_match = re.search(r'\[Perf\] Global GA: ([\d.]+)% \| Global Loss: ([\d.]+)', round_content)
            if old_perf_match:
                data.rounds.append(round_num)
                data.global_acc.append(float(old_perf_match.group(1)))
                data.global_loss.append(float(old_perf_match.group(2)))
                # Check for old personalized format
                old_personalized_match = re.search(
                    r'\[Perf\] Personalized PA: ([\d.]+)% \| Personalized Loss: ([\d.]+)',
                    round_content
                )
                if old_personalized_match:
                    data.has_personalized = True
                    data.personalized_acc.append(float(old_personalized_match.group(1)))
                    data.personalized_loss.append(float(old_personalized_match.group(2)))
                else:
                    data.personalized_acc.append(None)
                    data.personalized_loss.append(None)
            else:
                continue  # Skip this round if we can't parse it
        
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
    """Create comprehensive visualizations for all experiments in 2x2 grid style"""
    if not experiments:
        print("No experiment data to visualize")
        return
    
    # Color palette and line styles for different experiments (matching image style)
    experiment_styles = {
        'bl1': {'color': '#000000', 'linestyle': '-', 'marker': '+', 'label': 'BL1'},      # Solid black with +
        'bl2': {'color': '#FF0000', 'linestyle': '--', 'marker': '*', 'label': 'BL2'},     # Dashed red with *
        'bl3': {'color': '#0000FF', 'linestyle': ':', 'marker': 'o', 'label': 'BL3'},      # Dotted blue with •
        'abl1': {'color': '#00FF00', 'linestyle': '-.', 'marker': 'x', 'label': 'ABL1'},   # Dash-dot green with x
        'abl2': {'color': '#9467bd', 'linestyle': '-', 'marker': 's', 'label': 'ABL2'},    # Purple with square
    }
    
    # Create figure with 2x2 grid layout (clean style like the reference image)
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # 1. Average Test Accuracy (Top-Left)
    ax1 = fig.add_subplot(gs[0, 0])
    for exp_name, data in experiments.items():
        style = experiment_styles.get(exp_name, {'color': '#000000', 'linestyle': '-', 'marker': 'o', 'label': exp_name.upper()})
        ax1.plot(data.rounds, data.global_acc, 
                linestyle=style['linestyle'], marker=style['marker'], 
                label=style['label'], color=style['color'], 
                linewidth=2, markersize=6, markevery=max(1, len(data.rounds)//10))
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Average Test Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    # Set y-axis limit based on all data
    all_acc_values = [acc for data in experiments.values() for acc in data.global_acc if data.global_acc]
    if all_acc_values:
        ax1.set_ylim(bottom=max(0, min(all_acc_values) - 5))
    
    # 2. Average Loss (Top-Right)
    ax2 = fig.add_subplot(gs[0, 1])
    for exp_name, data in experiments.items():
        style = experiment_styles.get(exp_name, {'color': '#000000', 'linestyle': '-', 'marker': 'o', 'label': exp_name.upper()})
        ax2.plot(data.rounds, data.global_loss, 
                linestyle=style['linestyle'], marker=style['marker'], 
                label=style['label'], color=style['color'], 
                linewidth=2, markersize=6, markevery=max(1, len(data.rounds)//10))
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Average Loss', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 3. Cumulative Communication Cost (Bottom-Left)
    ax3 = fig.add_subplot(gs[1, 0])
    for exp_name, data in experiments.items():
        style = experiment_styles.get(exp_name, {'color': '#000000', 'linestyle': '-', 'marker': 'o', 'label': exp_name.upper()})
        ax3.plot(data.rounds, data.cumulative_data_mb, 
                linestyle=style['linestyle'], marker=style['marker'], 
                label=style['label'], color=style['color'], 
                linewidth=2, markersize=6, markevery=max(1, len(data.rounds)//10))
    ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Communication Cost (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Cumulative Communication Cost', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 4. Cumulative Time (Bottom-Right)
    ax4 = fig.add_subplot(gs[1, 1])
    for exp_name, data in experiments.items():
        style = experiment_styles.get(exp_name, {'color': '#000000', 'linestyle': '-', 'marker': 'o', 'label': exp_name.upper()})
        ax4.plot(data.rounds, data.cumulative_time_s, 
                linestyle=style['linestyle'], marker=style['marker'], 
                label=style['label'], color=style['color'], 
                linewidth=2, markersize=6, markevery=max(1, len(data.rounds)//10))
    ax4.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Cumulative Time', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='best', fontsize=10, framealpha=0.9)
    
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

