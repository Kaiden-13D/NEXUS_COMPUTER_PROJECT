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
    
    # Create figure with subplots - 추가된 통합 cost 차트를 위해 공간 확장
    if has_any_personalized:
        fig = plt.figure(figsize=(20, 18))
        gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    
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
        
        # 5. Round Communication Cost (Bar Chart)
        ax5 = fig.add_subplot(gs[2, 0])
        x_pos = np.arange(len(experiments))
        bar_width = 0.35
        # 마지막 라운드의 cost를 bar chart로 표시
        last_round_costs = []
        exp_names_list = list(experiments.keys())
        for exp_name in exp_names_list:
            if experiments[exp_name].round_cost_kb:
                last_round_costs.append(experiments[exp_name].round_cost_kb[-1])
            else:
                last_round_costs.append(0)
        bars = ax5.bar(x_pos, last_round_costs, bar_width, 
                      color=[colors.get(name, '#000000') for name in exp_names_list],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax5.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Last Round Communication Cost (KB)', fontsize=12, fontweight='bold')
        ax5.set_title('Round Communication Cost Comparison (Bar Chart)', fontsize=14, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([name.upper() for name in exp_names_list], fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, (bar, cost) in enumerate(zip(bars, last_round_costs)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.0f} KB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 6. Round Simulated Time (Bar Chart)
        ax6 = fig.add_subplot(gs[2, 1])
        last_round_times = []
        for exp_name in exp_names_list:
            if experiments[exp_name].round_time_s:
                last_round_times.append(experiments[exp_name].round_time_s[-1])
            else:
                last_round_times.append(0)
        bars = ax6.bar(x_pos, last_round_times, bar_width,
                      color=[colors.get(name, '#000000') for name in exp_names_list],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax6.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Last Round Simulated Time (seconds)', fontsize=12, fontweight='bold')
        ax6.set_title('Round Simulated Time Comparison (Bar Chart)', fontsize=14, fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([name.upper() for name in exp_names_list], fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, (bar, time) in enumerate(zip(bars, last_round_times)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
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
        
        # 9. 통합 Cost (Memory + Time) - Normalized Bar Chart
        ax9 = fig.add_subplot(gs[4, :])
        # 정규화된 통합 cost 계산 (α=1.0, β=0.1)
        alpha, beta = 1.0, 0.1
        max_comm = max([max(data.cumulative_data_mb) if data.cumulative_data_mb else 0 
                       for data in experiments.values()])
        max_time = max([max(data.cumulative_time_s) if data.cumulative_time_s else 0 
                       for data in experiments.values()])
        
        integrated_costs = []
        for exp_name in exp_names_list:
            data = experiments[exp_name]
            if data.cumulative_data_mb and data.cumulative_time_s:
                # 정규화된 통합 cost
                norm_comm = (data.cumulative_data_mb[-1] / max_comm) if max_comm > 0 else 0
                norm_time = (data.cumulative_time_s[-1] / max_time) if max_time > 0 else 0
                integrated_cost = alpha * norm_comm + beta * norm_time
            else:
                integrated_cost = 0
            integrated_costs.append(integrated_cost)
        
        # Stacked bar chart로 memory와 time을 함께 표시
        x_pos_wide = np.arange(len(exp_names_list))
        width = 0.6
        norm_comm_values = [(data.cumulative_data_mb[-1] / max_comm) if max_comm > 0 and data.cumulative_data_mb else 0 
                           for data in [experiments[name] for name in exp_names_list]]
        norm_time_values = [(data.cumulative_time_s[-1] / max_time) * beta if max_time > 0 and data.cumulative_time_s else 0 
                           for data in [experiments[name] for name in exp_names_list]]
        
        bars1 = ax9.bar(x_pos_wide, norm_comm_values, width, 
                       label='Normalized Comm Cost (α=1.0)', 
                       color=[colors.get(name, '#000000') for name in exp_names_list],
                       alpha=0.7, edgecolor='black', linewidth=1.2)
        bars2 = ax9.bar(x_pos_wide, norm_time_values, width, bottom=norm_comm_values,
                       label='Normalized Time Cost (β=0.1)', 
                       color=[colors.get(name, '#888888') for name in exp_names_list],
                       alpha=0.5, edgecolor='black', linewidth=1.2)
        
        ax9.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Normalized Integrated Cost', fontsize=12, fontweight='bold')
        ax9.set_title('Integrated Cost Comparison (Memory + Time) - Normalized Stacked Bar Chart', 
                     fontsize=14, fontweight='bold')
        ax9.set_xticks(x_pos_wide)
        ax9.set_xticklabels([name.upper() for name in exp_names_list], fontsize=10)
        ax9.legend(loc='upper left', fontsize=10)
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 통합 cost 값 표시
        for i, (bar1, bar2, cost) in enumerate(zip(bars1, bars2, integrated_costs)):
            total_height = bar1.get_height() + bar2.get_height()
            ax9.text(bar1.get_x() + bar1.get_width()/2., total_height,
                    f'Total: {cost:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        # Without personalized metrics, use 3x2 layout
        # 3. Round Communication Cost (Bar Chart)
        ax3 = fig.add_subplot(gs[1, 0])
        x_pos = np.arange(len(experiments))
        bar_width = 0.35
        exp_names_list = list(experiments.keys())
        last_round_costs = []
        for exp_name in exp_names_list:
            if experiments[exp_name].round_cost_kb:
                last_round_costs.append(experiments[exp_name].round_cost_kb[-1])
            else:
                last_round_costs.append(0)
        bars = ax3.bar(x_pos, last_round_costs, bar_width, 
                      color=[colors.get(name, '#000000') for name in exp_names_list],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax3.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Last Round Communication Cost (KB)', fontsize=12, fontweight='bold')
        ax3.set_title('Round Communication Cost Comparison (Bar Chart)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([name.upper() for name in exp_names_list], fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        for i, (bar, cost) in enumerate(zip(bars, last_round_costs)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.0f} KB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. Round Simulated Time (Bar Chart)
        ax4 = fig.add_subplot(gs[1, 1])
        last_round_times = []
        for exp_name in exp_names_list:
            if experiments[exp_name].round_time_s:
                last_round_times.append(experiments[exp_name].round_time_s[-1])
            else:
                last_round_times.append(0)
        bars = ax4.bar(x_pos, last_round_times, bar_width,
                      color=[colors.get(name, '#000000') for name in exp_names_list],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax4.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Last Round Simulated Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('Round Simulated Time Comparison (Bar Chart)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([name.upper() for name in exp_names_list], fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        for i, (bar, time) in enumerate(zip(bars, last_round_times)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
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
        
        # 7. 통합 Cost (Memory + Time) - Normalized Bar Chart
        ax7 = fig.add_subplot(gs[3, :])
        alpha, beta = 1.0, 0.1
        max_comm = max([max(data.cumulative_data_mb) if data.cumulative_data_mb else 0 
                       for data in experiments.values()])
        max_time = max([max(data.cumulative_time_s) if data.cumulative_time_s else 0 
                       for data in experiments.values()])
        
        integrated_costs = []
        for exp_name in exp_names_list:
            data = experiments[exp_name]
            if data.cumulative_data_mb and data.cumulative_time_s:
                norm_comm = (data.cumulative_data_mb[-1] / max_comm) if max_comm > 0 else 0
                norm_time = (data.cumulative_time_s[-1] / max_time) if max_time > 0 else 0
                integrated_cost = alpha * norm_comm + beta * norm_time
            else:
                integrated_cost = 0
            integrated_costs.append(integrated_cost)
        
        x_pos_wide = np.arange(len(exp_names_list))
        width = 0.6
        norm_comm_values = [(data.cumulative_data_mb[-1] / max_comm) if max_comm > 0 and data.cumulative_data_mb else 0 
                           for data in [experiments[name] for name in exp_names_list]]
        norm_time_values = [(data.cumulative_time_s[-1] / max_time) * beta if max_time > 0 and data.cumulative_time_s else 0 
                           for data in [experiments[name] for name in exp_names_list]]
        
        bars1 = ax7.bar(x_pos_wide, norm_comm_values, width, 
                       label='Normalized Comm Cost (α=1.0)', 
                       color=[colors.get(name, '#000000') for name in exp_names_list],
                       alpha=0.7, edgecolor='black', linewidth=1.2)
        bars2 = ax7.bar(x_pos_wide, norm_time_values, width, bottom=norm_comm_values,
                       label='Normalized Time Cost (β=0.1)', 
                       color=[colors.get(name, '#888888') for name in exp_names_list],
                       alpha=0.5, edgecolor='black', linewidth=1.2)
        
        ax7.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Normalized Integrated Cost', fontsize=12, fontweight='bold')
        ax7.set_title('Integrated Cost Comparison (Memory + Time) - Normalized Stacked Bar Chart', 
                     fontsize=14, fontweight='bold')
        ax7.set_xticks(x_pos_wide)
        ax7.set_xticklabels([name.upper() for name in exp_names_list], fontsize=10)
        ax7.legend(loc='upper left', fontsize=10)
        ax7.grid(True, alpha=0.3, axis='y')
        
        for i, (bar1, bar2, cost) in enumerate(zip(bars1, bars2, integrated_costs)):
            total_height = bar1.get_height() + bar2.get_height()
            ax7.text(bar1.get_x() + bar1.get_width()/2., total_height,
                    f'Total: {cost:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
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

