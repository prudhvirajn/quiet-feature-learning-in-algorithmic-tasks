#!/usr/bin/env python3
"""

Usage:
    python plot_scaling_law.py <base_directory> <output_directory> [options]

Arguments:
    base_directory: Path to the base directory containing experimental results
    output_directory: Path to the directory where plots will be saved

Options:
    --top-k: Number of best runs to display for each compute budget (default: 1)
    --pareto-line: Whether to draw Pareto frontier lines (default: True)
    --figsize: Figure size as "width,height" (default: "30,14")
    --dpi: DPI for saved figures (default: 300)
    --format: Output format (png, pdf, svg, etc.) (default: png)
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.ticker as mticker
import math
from matplotlib.lines import Line2D


def plot_phase_transitions_across_tasks(base_directory, output_directory, 
                                       figsize=(30, 14), top_k=1, pareto_line=True, 
                                       y_axis_left_only=False, dpi=300, format='png'):
    """
    Create plots showing min val loss, corresponding test accuracy, and corresponding train loss vs compute budget
    for different tasks and task lengths, demonstrating phase transitions in scaling laws.
    
    Parameters:
    -----------
    base_directory : str
        Base directory containing experimental results
    output_directory : str
        Directory to save the generated plots
    top_k : int
        Number of best runs to display for each compute budget based on validation loss.
    pareto_line : bool
        Whether to draw a line connecting the best points (Pareto frontier).
    y_axis_left_only : bool
        If True, only display y-axis labels for the leftmost subplots.
    dpi : int
        DPI for saved figures
    format : str
        Output format for saved figures
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Define tasks information with max_compute_budget, display names, random baselines, and min task length
    tasks_info = [
        # Row 1: Algorithmic tasks
        {'task': 'addition', 'display_name': 'Addition', 'max_compute_budget': 1e15, 'random_baseline': (- 18 / 19 ) * math.log(0.5), 'min_task_length': 0},
        {'task': 'majority_of_majority', 'display_name': 'Majority of\nmajorities', 'max_compute_budget': 1e15, 'random_baseline': (- 1 / 3 ) * math.log(0.5), 'min_task_length': 8},
        {'task': 'activity_selection', 'display_name': 'Activity\nselection', 'max_compute_budget': 1e15, 'random_baseline': -math.log(1 / 20), 'min_task_length': 0},
        {'task': 'multiplication', 'display_name': 'Multiplication', 'max_compute_budget': 1e18, 'random_baseline': (- 32 / 34 ) * math.log(0.5), 'min_task_length': 8},
        {'task': 'maximum_subarray', 'display_name': 'Maximum\nsubarray', 'max_compute_budget': 1e15, 'random_baseline': -math.log(1 / 20), 'min_task_length': 0},
        
        # Row 2: Graph tasks
        {'task': 'graph_breadth_first_search', 'display_name': 'Breadth first\nsearch', 'max_compute_budget': 1e17, 'random_baseline': -(11 / 13) * math.log(1 / 26), 'min_task_length': 0},
        {'task': 'graph_depth_first_search', 'display_name': 'Depth first\nsearch', 'max_compute_budget': 1e17, 'random_baseline': -(11 / 13) * math.log(1 / 26), 'min_task_length': 0},
        {'task': 'graph_min_spanning_tree_kruskal', 'display_name': 'Minimum\nspanning tree', 'max_compute_budget': 1e17, 'random_baseline': -math.log(1 / 27), 'min_task_length': 0},
        {'task': 'graph_path', 'display_name': 'Shortest\npath', 'max_compute_budget': 1e17, 'random_baseline': -math.log(1 / 27), 'min_task_length': 0},
        {'task': 'graph_topological_sort', 'display_name': 'Topological\nsorting', 'max_compute_budget': 1e17, 'random_baseline': -math.log(1 / 27), 'min_task_length': 0},
    ]
    
    # Set base directory for all tasks
    for task_info in tasks_info:
        task_info['base_directory'] = base_directory
    
    # Define metrics to plot
    metrics = [
        {'name': 'val_loss', 'title': 'Validation Loss', 'scale': 'log', 'label_prefix': 'Min '},
        {'name': 'test_acc_result', 'title': 'Test Accuracy', 'scale': 'linear', 'label_prefix': 'Corresponding '},
        # {'name': 'train_loss_step', 'title': 'Training Loss', 'scale': 'log', 'label_prefix': 'Corresponding '}
    ]
    
    # Function to get available task lengths for a given task
    def get_available_task_lengths(task_info):
        """Scan directories to find available task lengths for a task"""
        task_name = task_info['task']
        base_directory = task_info['base_directory']
        
        main_directory = os.path.join(
            base_directory,
            task_name,
            '1',
            'normal',
            '11111',
            '4',
            'input_reverseTrue',
            'output_reverseTrue'
        )
        
        task_lengths = []
        if os.path.exists(main_directory):
            for item in os.listdir(main_directory):
                item_path = os.path.join(main_directory, item)
                # Check if it's a directory and represents a task length (should be numeric)
                if os.path.isdir(item_path):
                    try:
                        task_length = int(item)
                        # Verify that individual_runs directory exists
                        individual_runs_path = os.path.join(item_path, 'individual_runs')
                        if os.path.exists(individual_runs_path):
                            task_lengths.append(task_length)
                    except ValueError:
                        continue
        
        return sorted(task_lengths)
    
    # Collect all unique task lengths across all tasks to create consistent styling
    all_unique_task_lengths = set()
    for task_info in tasks_info:
        available_lengths = get_available_task_lengths(task_info)
        min_task_length = task_info.get('min_task_length', 0)
        filtered_lengths = [length for length in available_lengths if length > min_task_length]
        all_unique_task_lengths.update(filtered_lengths)
    
    # Sort task lengths for consistent ordering
    all_unique_task_lengths = sorted(list(all_unique_task_lengths))
    
    # Define visual styles for different task lengths (consistent across all tasks)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Create a mapping from task length to visual style
    task_length_to_style = {}
    for i, task_length in enumerate(all_unique_task_lengths):
        task_length_to_style[task_length] = {
            'color': colors[i % len(colors)],
            'marker': markers[i % len(markers)],
            'line_style': line_styles[i % len(line_styles)]
        }
    
    # Create a figure for each metric
    for metric_idx, metric in enumerate(metrics):
        # Setup the figure and subplots
        fig, axes = plt.subplots(2, 5, figsize=figsize, squeeze=False)
        plt.subplots_adjust(hspace=2, wspace=0.7)
        
        # Define function to format x-axis ticks as powers of 10
        def format_power_of_ten(x, pos):
            """Format tick labels as powers of 10 with caret notation"""
            exponent = int(np.log10(x))
            return f'$10^{{{exponent}}}$'
        
        # For each task, plot the metric vs compute budget for all available task lengths
        for idx, task_info in enumerate(tasks_info):
            task_name = task_info['task']
            task_base_directory = task_info['base_directory']
            max_compute_budget = task_info['max_compute_budget']
            random_baseline = task_info.get('random_baseline', None)
            min_task_length = task_info.get('min_task_length', 0)  # Default to 0 if not specified
            
            # Get the corresponding subplot
            row = idx // 5
            col = idx % 5
            ax = axes[row][col]
            
            # Add border to subplot
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('black')
                spine.set_visible(True)
            
            # Get available task lengths for this task
            all_available_task_lengths = get_available_task_lengths(task_info)
            
            # Filter out task lengths that are <= min_task_length
            available_task_lengths = [length for length in all_available_task_lengths if length > min_task_length]
            
            if not available_task_lengths:
                print(f"No valid task lengths found for task {task_name} (after filtering with min_task_length={min_task_length})")
                display_name = task_info.get('display_name', task_name)
                ax.text(0.5, 0.5, f"No data for {display_name}\n(min length: {min_task_length})", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{display_name}", fontsize=12)
                continue
            
            print(f"Task {task_name}: Found task lengths {available_task_lengths} (filtered from {all_available_task_lengths} with min_task_length={min_task_length})")
            
            # Plot each task length with consistent visual style
            for task_length in available_task_lengths:
                # Get consistent visual style for this task length
                style = task_length_to_style[task_length]
                color = style['color']
                marker = style['marker']
                line_style = style['line_style']
                
                # Construct the full directory path
                main_directory = os.path.join(
                    task_base_directory,
                    task_name,
                    '1',
                    'normal',
                    '11111',
                    '4',
                    'input_reverseTrue',
                    'output_reverseTrue'
                )
                
                directory_path = os.path.join(
                    main_directory,
                    str(task_length),
                    'individual_runs'
                )
                
                if not os.path.isdir(directory_path):
                    print(f"Directory for task {task_name}, length {task_length} does not exist: {directory_path}")
                    continue
                
                # Dictionary to store top-k runs for each compute budget
                compute_to_top_k_metrics = defaultdict(list)
                
                # Process all compute budget subdirectories
                for dir_name in os.listdir(directory_path):
                    compute_budget_path = os.path.join(directory_path, dir_name)
                    
                    if not os.path.isdir(compute_budget_path):
                        continue
                        
                    try:
                        compute_budget = int(dir_name)
                    except ValueError:
                        print(f"Skipping non-integer compute budget directory: {compute_budget_path}")
                        continue
                    
                    # Skip if compute budget exceeds max
                    if compute_budget > max_compute_budget:
                        continue
                    
                    # List to collect all runs for this compute budget
                    all_runs = []
                    
                    # Process all JSON files in this compute budget directory
                    for file in os.listdir(compute_budget_path):
                        if file.endswith('.json'):
                            file_path = os.path.join(compute_budget_path, file)
            
                            try:
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                            except json.JSONDecodeError:
                                print(f"Skipping invalid JSON file: {file_path}")
                                continue

                            # Extract metrics
                            summary = data.get('summary', {})
                            val_loss = summary.get('val_loss')
                            test_acc = summary.get('test_acc_result')
                            train_loss = summary.get('train_loss_step')
                            epochs = summary.get('epoch')
                            
                            # Only consider runs where epoch is 1 and all metrics are available
                            if epochs == 1 and val_loss is not None:
                                # Add this run to the list
                                all_runs.append({
                                    'val_loss': val_loss,
                                    'test_acc_result': test_acc,
                                    'train_loss_step': train_loss
                                })
                    
                    # Sort runs by validation loss and keep only top-k
                    if all_runs:
                        sorted_runs = sorted(all_runs, key=lambda x: x['val_loss'])
                        compute_to_top_k_metrics[compute_budget] = sorted_runs[:top_k]
                
                # Prepare data for plotting
                compute_budgets = []
                metric_values = []
                alpha_values = []  # For opacity based on ranking
                
                # For each compute budget, get the top-k values
                for compute in sorted(compute_to_top_k_metrics.keys()):
                    for rank, run_metrics in enumerate(compute_to_top_k_metrics[compute]):
                        compute_budgets.append(compute)
                        # Get the value for the current metric from this run
                        metric_values.append(run_metrics.get(metric['name']))
                        # Calculate alpha (transparency) based on rank (best run = most opaque)
                        alpha_values.append(0.9 - 0.5 * (rank / top_k) if top_k > 1 else 0.8)
                
                if not compute_budgets or not metric_values:
                    print(f"No valid data found for task {task_name} with task length {task_length}.")
                    continue
                
                # Plot metric value vs compute budget with varying opacity
                for i in range(len(compute_budgets)):
                    ax.scatter(compute_budgets[i], metric_values[i], color=color, marker=marker, 
                               s=80 * 4, edgecolors='w', alpha=alpha_values[i])
                
                # Add Pareto frontier line if requested
                if pareto_line:
                    # Find best (minimum) value for each unique compute budget
                    pareto_points = {}
                    for cb, mv in zip(compute_budgets, metric_values):
                        if cb not in pareto_points or mv < pareto_points[cb]:
                            pareto_points[cb] = mv
                    
                    # Sort by compute budget for line plotting
                    pareto_x = sorted(pareto_points.keys())
                    pareto_y = [pareto_points[x] for x in pareto_x]
                    
                    # Plot the Pareto frontier line
                    ax.plot(pareto_x, pareto_y, color=color, linestyle=line_style, 
                           linewidth=2 * 2, alpha=0.8)
                
                print(f"{task_name} (length {task_length}): Plotting {len(metric_values)} points for {len(set(compute_budgets))} compute budgets")
            
            # Add horizontal dotted line for random baseline if the metric is val_loss
            if random_baseline is not None and metric['name'] == 'val_loss':
                # Calculate the axis limits to ensure the line spans the full width
                x_min, x_max = ax.get_xlim()
                ax.hlines(random_baseline, x_min, x_max, colors='red', linestyles='dashed', 
                         linewidths=1.5 * 4, alpha=0.7)
            
            # Set scales
            ax.set_xscale('log')
            ax.set_yscale(metric['scale'])
            ax.grid(False)

            ax.tick_params(axis='both', which='major', labelsize=36, width=2, length=6)
            ax.tick_params(axis='both', which='minor', width=1.5, length=4)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set x-axis ticks to be exactly the unique compute budgets we have data for
            all_budgets = []
            for task_length in available_task_lengths:
                directory_path = os.path.join(
                    task_base_directory, task_name, '1', 'normal', '11111', '4',
                    'input_reverseTrue', 'output_reverseTrue', str(task_length), 'individual_runs'
                )
                if os.path.isdir(directory_path):
                    for dir_name in os.listdir(directory_path):
                        try:
                            budget = int(dir_name)
                            if budget <= max_compute_budget:
                                all_budgets.append(budget)
                        except ValueError:
                            continue
            
            if all_budgets:
                unique_budgets = sorted(set(all_budgets))
                unique_budgets = [item for idx, item in enumerate(unique_budgets) if idx % 2 == 0]
                ax.set_xticks(unique_budgets)
                
                # Apply the power of 10 formatter for x-axis
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_power_of_ten))
            
            # Set title
            display_name = task_info.get('display_name', task_name)
            ax.set_title(f"{display_name}", fontsize=42)
        
        # Global labels
        fig.text(0.5, 0.1, 'Training FLOPs', ha='center', fontsize=52)
        fig.text(0.02, 0.45, metric['title'], va='center', rotation='vertical', fontsize=52)

        # Create combined legend with task lengths and random baseline
        combined_legend_elements = []
        
        # Add task length legend elements if there are multiple task lengths
        if len(all_unique_task_lengths) > 1:
            for task_length in all_unique_task_lengths:
                style = task_length_to_style[task_length]
                combined_legend_elements.append(
                    Line2D([0], [0], color=style['color'], marker=style['marker'], 
                           linestyle=style['line_style'], markersize=8, linewidth=2, 
                           label=f'Length {task_length}')
                )
        
        # Add random baseline legend element
        combined_legend_elements.append(
            Line2D([0], [0], color='red', linestyle='dashed', lw=1.5 * 4, alpha=0.7, label='Random')
        )
        
        fig.legend(handles=combined_legend_elements, 
                    loc='lower center', 
                    bbox_to_anchor=(0.5, 0.01), 
                    ncol=len(combined_legend_elements), 
                    fontsize=24)
        
        # Add a title for the entire figure
        plt.suptitle(f"Scaling Law Estimation", fontsize=72, y=0.98)
        
        # Adjust layout
        plt.tight_layout(h_pad=4, w_pad=0, rect=[0.04, 0.15, 1, 0.99])
        
        # Save the figure
        metric_name = metric['name'].replace('_', '-')
        output_filename = f"scaling_law_{metric_name}.{format}"
        output_path = os.path.join(output_directory, output_filename)
        
        plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot: {output_path}")
        
        # Close the figure to free memory
        plt.close(fig)


def parse_figsize(figsize_str):
    """Parse figsize string like '30,14' into tuple (30, 14)"""
    try:
        width, height = map(float, figsize_str.split(','))
        return (width, height)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid figsize format: {figsize_str}. Expected 'width,height'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate phase transition plots from experimental results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_phase_transitions.py /path/to/results /path/to/output
    python plot_phase_transitions.py /data/experiments /plots --top-k 3 --figsize 20,10
    python plot_phase_transitions.py ./results ./figures --format pdf --dpi 600
        """
    )
    
    parser.add_argument('base_directory', 
                       help='Base directory containing experimental results')
    parser.add_argument('output_directory',
                       help='Directory to save the generated plots')
    parser.add_argument('--top-k', type=int, default=1,
                       help='Number of best runs to display for each compute budget (default: 1)')
    parser.add_argument('--pareto-line', action='store_true', default=True,
                       help='Draw Pareto frontier lines (default: True)')
    parser.add_argument('--no-pareto-line', dest='pareto_line', action='store_false',
                       help='Do not draw Pareto frontier lines')
    parser.add_argument('--figsize', type=parse_figsize, default=(30, 14),
                       help='Figure size as "width,height" (default: "30,14")')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 300)')
    parser.add_argument('--format', default='png',
                       choices=['png', 'pdf', 'svg', 'eps', 'ps', 'tiff'],
                       help='Output format for saved figures (default: png)')
    parser.add_argument('--y-axis-left-only', action='store_true', default=False,
                       help='Only display y-axis labels for leftmost subplots')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.base_directory):
        print(f"Error: Base directory does not exist: {args.base_directory}")
        sys.exit(1)
    
    if not os.path.isdir(args.base_directory):
        print(f"Error: Base directory is not a directory: {args.base_directory}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_directory, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create output directory {args.output_directory}: {e}")
        sys.exit(1)
    
    print(f"Base directory: {args.base_directory}")
    print(f"Output directory: {args.output_directory}")
    print(f"Parameters: top_k={args.top_k}, pareto_line={args.pareto_line}, "
          f"figsize={args.figsize}, dpi={args.dpi}, format={args.format}")
    
    # Generate plots
    try:
        plot_phase_transitions_across_tasks(
            base_directory=args.base_directory,
            output_directory=args.output_directory,
            figsize=args.figsize,
            top_k=args.top_k,
            pareto_line=args.pareto_line,
            y_axis_left_only=args.y_axis_left_only,
            dpi=args.dpi,
            format=args.format
        )
        print("Plot generation completed successfully!")
    except Exception as e:
        print(f"Error during plot generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()