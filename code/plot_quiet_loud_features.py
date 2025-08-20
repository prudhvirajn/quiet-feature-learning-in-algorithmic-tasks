#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
from matplotlib.patches import Patch
from scipy import stats
import argparse
import os

# Define standard colors globally
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default colors: blue, orange, green

def plot_metrics(pickle_file, features_metrics, ax=None, positions=None, 
                x0=int(1e10), x1=int(1e11), title=None, xlabel="FLOPs", ylabel=None, 
                show_grid=True, position_labels=None, show_ylabel=True, position_grouping=None,
                phase_style="gray_area", subplot_index=0, markersize=16, tst_fontsize=14 * 2, vertical_linewidth=2.5, horizontal_linewidth=4):
    """
    Plot metrics for given features across token positions or as averages.
    
    Parameters:
    -----------
    pickle_file : str
        Path to the pickle file containing metrics data
    features_metrics : list of dict
        Each dict should have keys:
        - 'feature': The feature/attribute name in the data
        - 'metric': The metric name in the data
        - 'label': Display name for the legend (optional)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, returns the data for external plotting
    positions : list of int, optional
        Token positions to plot. If None, computes the average across all positions
    position_grouping : str, optional
        If set to "thirds", will group positions into first 1/3, middle 1/3, and last 1/3 and average them
    x0, x1 : int, optional
        Range for highlighting regime change
    title : str, optional
        Title for the plot
    xlabel : str, optional
        Label for the x-axis
    ylabel : str, optional
        Label for the y-axis
    show_grid : bool, optional
        Whether to show grid lines
    position_labels : list of str, optional
        Custom labels for positions (e.g., ["Beginning", "Middle", "End"])
    show_ylabel : bool, optional
        Whether to show the y-axis label
    phase_style : str, optional
        Style for phase visualization: "gray_area", "red_line", or "phase_labels"
    subplot_index : int, optional
        Index of the current subplot (used to show annotations only on first subplot)
        
    Returns:
    --------
    matplotlib.axes.Axes or dict
        Axes with plot or data for external plotting
    """
    # Load metrics from pickle file
    with open(pickle_file, "rb") as f:
        overall_metrics = pickle.load(f)
    
    # Store data for each feature-metric pair
    data_dict = {}
    
    # Process each feature-metric pair
    for fm in features_metrics:
        feature_name = fm['feature']
        metric_name = fm['metric']
        display_label = fm.get('label', f"{feature_name}")
        
        # New: We need both train and test metrics
        train_metric = metric_name.replace('test_', 'train_')
        test_metric = metric_name if 'test_' in metric_name else 'test_' + metric_name
        
        # Dictionaries to hold data
        # For each position, we'll store module info with both train and test metrics
        token_position_module_data = {}  # For budgets > 0
        baseline_position_data = {}  # For random baseline (budget 0)
        
        random_counter = 0
        
        # Process each compute budget
        for budget_key, budget_entry in overall_metrics.items():
            budget_value = budget_key
            metrics_dict = budget_entry.get("metrics", {})
            
            if feature_name not in metrics_dict:
                continue
            feature_metrics = metrics_dict[feature_name]
            
            if 'random' not in str(budget_key):
                budget_value = float(budget_value)
                random_budget = False
            else:
                budget_value = random_counter
                random_counter += 1
                random_budget = True
            
            # Extract metrics data from layers and modules
            for layer_key, module_dict in feature_metrics.items():
                for module_key, module_metrics in module_dict.items():
                    # Skip if either train or test metric is missing
                    if train_metric not in module_metrics or test_metric not in module_metrics:
                        continue
                    
                    # Get train and test metric arrays
                    train_arr = np.array(module_metrics[train_metric]).flatten()
                    test_arr = np.array(module_metrics[test_metric]).flatten()
                    
                    # Make sure arrays are the same length
                    if len(train_arr) != len(test_arr):
                        continue
                    
                    num_positions = len(train_arr)
                    
                    for pos in range(num_positions):
                        if random_budget:
                            if pos not in baseline_position_data:
                                baseline_position_data[pos] = {}
                            if budget_value not in baseline_position_data[pos]:
                                baseline_position_data[pos][budget_value] = []
                            baseline_position_data[pos][budget_value].append(test_arr[pos])
                        else:
                            if pos not in token_position_module_data:
                                token_position_module_data[pos] = {}
                            if budget_value not in token_position_module_data[pos]:
                                token_position_module_data[pos][budget_value] = []
                            
                            # Store both train and test metrics for each module
                            token_position_module_data[pos][budget_value].append({
                                'module': f"{layer_key}_{module_key}",
                                'train_metric': train_arr[pos],
                                'test_metric': test_arr[pos]
                            })
        
        # Process data for each position
        token_position_aggregated = {}
        for pos, budget_dict in token_position_module_data.items():
            token_position_aggregated[pos] = []
            
            for budget_value, module_metrics in budget_dict.items():
                if not module_metrics:
                    continue
                
                # 1. Find the module with the best (minimum) train metric
                best_module = min(module_metrics, key=lambda x: x['train_metric'])
                
                # 2. Report that module's test metric
                test_value = best_module['test_metric']
                
                token_position_aggregated[pos].append((budget_value, test_value))
            
            # Sort by budget
            token_position_aggregated[pos].sort(key=lambda x: x[0])
        
        # Aggregate values for random baseline per token position
        baseline_aggregated = {}
        for pos, budget_dict in baseline_position_data.items():
            data = []
            baseline_aggregated[pos] = []
            for budget_value, values in budget_dict.items():
                agg_value = min(values)  # Use min for loss metrics
                data.append(agg_value)
            baseline_aggregated[pos].append(stats.mstats.gmean(data))  # Use gmean instead of mean
        
        # If using position_grouping="thirds", group positions into thirds and average

        if position_grouping == "thirds":
            # Get all available positions from token_position_aggregated
            all_positions = sorted(token_position_aggregated.keys())
            
            if not all_positions:
                continue
                
            # Divide positions into three groups
            n_positions = len(all_positions)
            first_third = all_positions[:n_positions//3]
            middle_third = all_positions[n_positions//3:2*n_positions//3]
            last_third = all_positions[2*n_positions//3:]
            
            position_groups = {
                'beginning': first_third,
                'middle': middle_third,
                'end': last_third
            }
            
            # Create data structure for the grouped positions
            grouped_data = {}
            grouped_baseline = {}
            
            # Average within each group
            for group_name, group_positions in position_groups.items():
                # Skip if group is empty
                if not group_positions:
                    continue
                
                # Get all unique budgets across positions in this group
                all_budgets = set()
                for pos in group_positions:
                    if pos in token_position_aggregated:
                        all_budgets.update(budget for budget, _ in token_position_aggregated[pos])
                
                # For each budget, average across positions
                grouped_data[group_name] = []
                for budget in sorted(all_budgets):
                    # Collect values for this budget across positions
                    values = []
                    for pos in group_positions:
                        if pos in token_position_aggregated:
                            for b, val in token_position_aggregated[pos]:
                                if b == budget:
                                    values.append(val)

                    if budget < int(2 ** 12):
                        print(feature_name, values)
                    
                    # Average if we have values
                    if values:
                        avg_value = stats.mstats.gmean(values) # np.mean(values)
                        grouped_data[group_name].append((budget, avg_value))
                
                # Sort by budget
                grouped_data[group_name].sort(key=lambda x: x[0])
                
                # Also average baseline values for this group
                baseline_values = []
                for pos in group_positions:
                    if pos in baseline_aggregated:
                        baseline_values.extend(baseline_aggregated[pos])
                
                if baseline_values:
                    grouped_baseline[group_name] = stats.mstats.gmean(baseline_values)
            
            # Store grouped data
            data_dict[(feature_name, metric_name, display_label)] = {
                'grouped_data': grouped_data,
                'grouped_baseline': grouped_baseline
            }
        elif positions is not None:
            # Filter positions that exist in the data
            valid_positions = [pos for pos in positions if pos in token_position_aggregated]
            
            # Store position-specific data
            data_dict[(feature_name, metric_name, display_label)] = {
                'position_data': {pos: token_position_aggregated[pos] for pos in valid_positions},
                'baseline_data': {pos: baseline_aggregated[pos][0] for pos in valid_positions if pos in baseline_aggregated}
            }
        else:
            # Compute average across all positions
            unique_budgets = set()
            for pos, data_points in token_position_aggregated.items():
                for budget, _ in data_points:
                    unique_budgets.add(budget)
            unique_budgets = sorted(unique_budgets)
            
            budget_avg = {}
            for pos, data_points in token_position_aggregated.items():
                for budget, value in data_points:
                    if budget not in budget_avg:
                        budget_avg[budget] = []
                    budget_avg[budget].append(value)
            
            for budget in budget_avg:
                budget_avg[budget] = stats.mstats.gmean(budget_avg[budget])
            
            # Compute overall average random baseline
            if baseline_aggregated:
                baseline_avg = stats.mstats.gmean([val[0] for val in baseline_aggregated.values()])
            else:
                baseline_avg = None
            
            # Store average data
            data_dict[(feature_name, metric_name, display_label)] = {
                'average_data': sorted([(budget, value) for budget, value in budget_avg.items()], key=lambda x: x[0]),
                'baseline_avg': baseline_avg
            }
    
    # If no axes provided, return the data for external plotting
    if ax is None:
        return data_dict
    
    # Plot on the provided axes
    
    # Use custom position labels if provided
    if position_labels is None and position_grouping == "thirds":
        position_labels = {"beginning": "Beginning", "middle": "Middle", "end": "End"}
    elif position_labels is None and positions is not None:
        position_labels = {pos: f"Position {pos}" for pos in positions}
    elif position_labels is not None and positions is not None:
        if len(position_labels) == len(positions):
            position_labels = {pos: label for pos, label in zip(positions, position_labels)}
        else:
            # Fall back to default if lengths don't match
            position_labels = {pos: f"Position {pos}" for pos in positions}
    
    # Plot data
    for idx, ((feature_name, metric_name, display_label), data) in enumerate(data_dict.items()):
        if 'grouped_data' in data:
            # For grouped position data, use different colors for each group
            for i, (group_name, data_points) in enumerate(data['grouped_data'].items()):
                if not data_points:
                    continue
                budgets, agg_values = zip(*data_points)
                
                # Use group name as label
                if position_labels and group_name in position_labels:
                    label = position_labels[group_name]
                else:
                    label = group_name.capitalize()
                
                # Use different markers for different groups
                markers = ["o", "s", "^", "d", "x"]  # circle, square, triangle, diamond, x
                ax.plot(budgets, agg_values, marker=markers[i % len(markers)], linewidth=2, ms=markersize, color=COLORS[i % len(COLORS)], label=label)
            
            # Add baseline lines with matching colors for each group
            for i, (group_name, baseline_value) in enumerate(data['grouped_baseline'].items()):
                # Use the same color as the corresponding group line
                matching_color = COLORS[i % len(COLORS)]
                ax.axhline(y=baseline_value, color=matching_color, linestyle=":", linewidth=horizontal_linewidth, alpha=0.5)
        elif 'position_data' in data:
            # For position-specific data, use different colors for each position
            for i, (pos, data_points) in enumerate(data['position_data'].items()):
                if not data_points:
                    continue
                budgets, agg_values = zip(*data_points)
                
                # Use just position labels if available
                if position_labels and pos in position_labels:
                    label = position_labels[pos]
                else:
                    label = f"Position {pos}"
                
                # Use different markers for different positions
                markers = ["o", "s", "^", "d", "x"]  # circle, square, triangle, diamond, x
                ax.plot(budgets, agg_values, marker=markers[i % len(markers)], linewidth=2, ms=8, color=COLORS[i % len(COLORS)], label=label)
            
            # Add baseline lines with matching colors for each position
            for i, (pos, baseline_value) in enumerate(data['baseline_data'].items()):
                # Use the same color as the corresponding position line
                matching_color = COLORS[i % len(COLORS)]
                ax.axhline(y=baseline_value, color=matching_color, linestyle=":", linewidth=horizontal_linewidth, alpha=0.5)
        else:
            # Plot average data
            if data['average_data']:
                budgets, avg_values = zip(*data['average_data'])
                ax.plot(budgets, avg_values, marker="o", label=display_label)
            
            # Add the overall random baseline if available, using the main line color
            if data['average_data'] and data['baseline_avg'] is not None:
                # Use the same color as the main line
                main_color = COLORS[0]  # Use first color for average data
                ax.axhline(y=data['baseline_avg'], color=main_color, linestyle=":", alpha=0.5)
    
    # Add common elements
    ax.set_xscale("log")
    ax.set_yscale("log")
    if title:
        ax.set_title(title, fontweight='bold')
    
    # Show grid lines based on parameter
    ax.grid(False)
    
    # Add phase visualization based on the chosen style
    if phase_style == "gray_area":
        # Original gray area style
        ax.axvspan(x0, x1, color='gray', alpha=0.3)
    elif phase_style == "red_line":
        # Suggestion 1: Red vertical line with arrow and label
        # Only show on first subplot
        if subplot_index == 0:
            ax.axvline(x=x1, color='red', linestyle='--', linewidth=vertical_linewidth, alpha=0.8)
            
            # Add annotation with arrow
            ylim = ax.get_ylim()
            # Position at 60% height to avoid baseline overlap
            y_pos = 10**(np.log10(ylim[0]) + 0.4 * (np.log10(ylim[1]) - np.log10(ylim[0])))
            
            ax.annotate('Task Success\nThreshold', 
                       xy=(x1, y_pos), 
                       xytext=(x1 * 0.1, y_pos),  # Place text to the left of the line
                       fontsize=tst_fontsize,
                       ha='right',
                       va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        else:
            # For other subplots, just show the line without annotation
            ax.axvline(x=x1, color='red', linestyle='--', linewidth=vertical_linewidth, alpha=0.8)
    elif phase_style == "phase_labels":
        # Suggestion 2: Two phase regions with labels
        # Add subtle background colors
        ax.axvspan(ax.get_xlim()[0], x0, color='lightblue', alpha=0.15)
        ax.axvspan(x0, ax.get_xlim()[1], color='lightcoral', alpha=0.15)
        
        # Only add phase labels on first subplot
        if subplot_index == 0:
            # Add phase labels at the bottom of the plot to avoid overlap with lines
            ylim = ax.get_ylim()
            y_label_pos = 10**(np.log10(ylim[0]) + 0.15 * (np.log10(ylim[1]) - np.log10(ylim[0])))
            
            # Calculate x positions for labels (center of each phase)
            xlim = ax.get_xlim()
            slow_x = np.sqrt(xlim[0] * x0)  # Geometric mean for log scale
            fast_x = np.sqrt(x0 * xlim[1])
            
            ax.text(slow_x, y_label_pos, 'Slow Phase', 
                   ha='center', va='bottom', fontsize=16, fontweight='bold', 
                   color='darkblue', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            ax.text(fast_x, y_label_pos, 'Fast Phase', 
                   ha='center', va='bottom', fontsize=16, fontweight='bold', 
                   color='darkred', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add vertical separator
        ax.axvline(x=x0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    return ax


def plot_multi_task_metrics(task_configs, figsize=(20, 5), nrows=1, ncols=None, 
                          show_grid=True, phase_transition_label="Phase Transition",
                          random_baseline_label="Random Baseline", x_label="Training FLOPs", 
                          w_pad=2, rect=[0, 0.07, 1, 0.92], super_title='Quiet Features', 
                          test_loc=(0.02, 0.47), x_loc=(0.5, 0.07), super_title_pos=(0.5, 0.98), 
                          bbox_to_anchor=(0.5, 0.02), legend_fontsize=18 * 2,
                          phase_style="gray_area", suptitle_fontsize=92, x_label_fontsize=52, y_label_fontsize=52, axes_title_fontsize=30, axis_tick_params_labelsize=24, markersize=16, tst_fontsize=14 * 2, vertical_linewidth=2.5, horizontal_linewidth=4):
    """
    Plot metrics for multiple tasks in a grid layout.
    
    Parameters:
    -----------
    task_configs : list of dict
        Configuration for each task plot. Each dict should contain:
        - pickle_file: Path to the pickle file
        - features_metrics: List of dicts with 'feature', 'metric', and optional 'label' keys
        - positions: (optional) List of token positions to plot
        - position_grouping: (optional) If "thirds", will group positions into thirds and average them
        - x0, x1: (optional) Range for highlighting regime change
        - title: (optional) Title for the plot
        - ylabel: (optional) Label for the y-axis
    figsize : tuple, optional
        Figure size (width, height)
    nrows : int, optional
        Number of rows in the grid
    ncols : int, optional
        Number of columns in the grid
    show_grid : bool, optional
        Whether to show grid lines
    phase_transition_label : str, optional
        Label for the phase transition in the legend
    random_baseline_label : str, optional
        Label for the random baseline in the legend
    phase_style : str, optional
        Style for phase visualization: "gray_area", "red_line", or "phase_labels"
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing all plots
    """
    num_tasks = len(task_configs)
    
    if ncols is None:
        ncols = math.ceil(num_tasks / nrows)
    
    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig)
    
    # Create axes for the plots
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j < num_tasks:
                axes.append(fig.add_subplot(gs[i, j]))
            else:
                break
    
    # Standard position labels
    default_position_labels = ["Beginning", "Middle", "End"]
    
    # Plot each task
    for i, task_config in enumerate(task_configs):
        if i >= len(axes):
            print(f"Warning: Only plotting the first {len(axes)} tasks.")
            break
        
        # Extract parameters from task configuration
        pickle_file = task_config['pickle_file']
        features_metrics = task_config['features_metrics']
        positions = task_config.get('positions', None)
        position_grouping = task_config.get('position_grouping', None)
        x0 = task_config.get('x0', int(1e10))
        x1 = task_config.get('x1', int(1e11))
        title = task_config.get('title', None)
        ylabel = task_config.get('ylabel', None)
        position_labels = task_config.get('position_labels', default_position_labels)
        
        # Only show y-axis label on the leftmost subplot if specified
        show_ylabel = i % ncols == 0
        
        # Create plot
        plot_metrics(
            pickle_file=pickle_file,
            features_metrics=features_metrics,
            ax=axes[i],
            positions=positions,
            position_grouping=position_grouping,
            x0=x0,
            x1=x1,
            title=title,
            xlabel="",  # Empty x-label since we'll use a common one
            ylabel=ylabel,
            show_grid=show_grid,
            position_labels=position_labels,
            show_ylabel=show_ylabel,
            phase_style=phase_style,  # Pass through the phase style
            subplot_index=i,  # Pass the subplot index,
            markersize=markersize,
            tst_fontsize=tst_fontsize,
            vertical_linewidth=vertical_linewidth,
            horizontal_linewidth=horizontal_linewidth
        )
        
        # Set title font size
        if title:
            axes[i].set_title(title, fontsize=axes_title_fontsize)
        
        # Set y-label font size
        if ylabel and show_ylabel:
            axes[i].set_ylabel(ylabel, fontsize=24)

        axes[i].tick_params(axis='both', which='major', labelsize=axis_tick_params_labelsize, width=2, length=6)
        axes[i].tick_params(axis='both', which='minor', width=1.5, length=4)
        
        # Remove top and right spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    
    # Collect handles and labels from the first subplot for common legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Add phase transition to legend based on style
    if phase_style == "gray_area":
        phase_handle = Patch(facecolor='gray', alpha=0.3)
        handles.append(phase_handle)
        labels.append(phase_transition_label)
    elif phase_style == "red_line":
        # Add red dashed line to legend
        import matplotlib.lines as mlines
        phase_handle = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2.5)
        # handles.append(phase_handle)
        # labels.append("Task Success Threshold")
    elif phase_style == "phase_labels":
        # Add phase indicators to legend
        slow_handle = Patch(facecolor='lightblue', alpha=0.15, edgecolor='darkblue', linewidth=2)
        fast_handle = Patch(facecolor='lightcoral', alpha=0.15, edgecolor='darkred', linewidth=2)
        handles.extend([slow_handle, fast_handle])
        labels.extend(["Slow Phase", "Fast Phase"])
    
    # Add a random baseline entry with the first color
    if handles:  # If there are position handles
        random_handle = plt.Line2D([0], [0], color=COLORS[0], linestyle=':', linewidth=2.5 * 2, alpha=1)
        handles.append(random_handle)
        labels.append(random_baseline_label)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add common x-axis label at the bottom of the figure
    plt.suptitle(f"{super_title}", fontsize=suptitle_fontsize, x=super_title_pos[0], y=super_title_pos[1])
    fig.text(x_loc[0], x_loc[1], x_label, ha='center', fontsize=x_label_fontsize)
    fig.text(test_loc[0], test_loc[1], 'Test Loss', va='center', rotation='vertical', fontsize=y_label_fontsize)

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=bbox_to_anchor, 
               ncol=len(handles), fontsize=legend_fontsize, frameon=True, framealpha=0.8)
    
    # Adjust layout to accommodate the common legend at the top
    plt.tight_layout(w_pad=0, rect=rect)
    
    return fig


def create_task_configs(input_dir):
    """Create task configurations with paths relative to input directory."""
    
    # Task configurations for "Quiet Features" plot
    quiet_task_configs = [
        {
            'pickle_file': os.path.join(input_dir, 'addition16_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'retrieve_first_op', 
                    'metric': 'test_log_loss',
                    'label': 'first\_operand'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e11),
            'x1': int(1e12),
            'title': 'Addition\n $\\mathit{first\_operand}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'addition16_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'carry', 
                    'metric': 'test_log_loss',
                    'label': 'carry'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e11),
            'x1': int(1e12),
            'title': 'Addition\n $\\mathit{carry}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'graph_breadth_first_search11_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'queue', 
                    'metric': 'test_log_loss',
                    'label': 'queue'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e12),
            'x1': int(1e13),
            'title': 'Breadth first search\n $\\mathit{queue}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'multiplication16_feature_metrics_first_operand.pickle'),
            'features_metrics': [
                {
                    'feature': 'retrieve_first_op', 
                    'metric': 'test_log_loss',
                    'label': 'first_operand'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e15),
            'x1': int(1e16),
            'title': 'Multiplication\n $\\mathit{first\_operand}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'multiplication16_feature_metrics_carry.pickle'),
            'features_metrics': [
                {
                    'feature': 'carry_16', 
                    'metric': 'test_log_loss',
                    'label': 'carry'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e15),
            'x1': int(1e16),
            'title': 'Multiplication\n $\\mathit{carry}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'maximum_subarray64_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'compute_prev_sign',
                    'metric': 'test_log_loss',
                    'label': 'Previous Integer Sign'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e12),
            'x1': int(1e13),
            'title': 'Maximum subarray\n $\\mathit{is\_prev\_negative}$',
            'ylabel': ''
        },
    ]
    
    # Task configurations for "Loud Features" plot
    loud_task_configs = [
        {
            'pickle_file': os.path.join(input_dir, 'graph_depth_first_search11_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'adjacency_list', 
                    'metric': 'test_log_loss',
                    'label': 'Adjacency List'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e13),
            'x1': int(1e14),
            'title': 'Depth first search\n $\\mathit{adjacency\_list}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'activity_selection16_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'retrieve_start_times',
                    'metric': 'test_mse',
                    'label': 'Start Times'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e12),
            'x1': int(1e13),
            'title': 'Activity selection\n $\\mathit{retrieve\_start\_times}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'maximum_subarray64_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'intermediate_sums',
                    'metric': 'test_mse',
                    'label': 'Intermediate Sum'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e12),
            'x1': int(1e13),
            'title': 'Maximum subarray\n $\\mathit{max\_ending\_here}$',
            'ylabel': ''
        },
        {
            'pickle_file': os.path.join(input_dir, 'graph_breadth_first_search11_feature_metrics.pickle'),
            'features_metrics': [
                {
                    'feature': 'adjacency_list', 
                    'metric': 'test_log_loss',
                    'label': 'Adjacency List'
                },
            ],
            'position_grouping': "thirds",
            'position_labels': ["Beginning", "Middle", "End"],
            'x0': int(1e12),
            'x1': int(1e13),
            'title': 'Breadth first search\n $\\mathit{adjacency\_list}$',
            'ylabel': ''
        },
    ]
    
    return quiet_task_configs, loud_task_configs


def main():
    parser = argparse.ArgumentParser(description='Generate metric plots from pickle files')
    parser.add_argument('input_dir', type=str, help='Directory containing pickle files')
    parser.add_argument('output_dir', type=str, help='Directory to save output plots')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved plots (default: 300)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create task configurations
    quiet_configs, loud_configs = create_task_configs(args.input_dir)
    
    print("Creating Quiet Features plot...")
    
    # Create "Quiet Features" plot
    fig1 = plot_multi_task_metrics(
        quiet_configs,
        figsize=(20, 12),
        nrows=2, 
        ncols=3,
        show_grid=False,
        phase_transition_label="Phase Transition",
        random_baseline_label="Random Baseline",
        x_label="Training FLOPs",
        rect=[0.04, 0.07, 1, 0.97],
        x_loc=(0.5, 0.03),
        bbox_to_anchor=(0.5, 0.02),
        w_pad=0,
        phase_style='red_line',
        super_title='Quiet Features',
        legend_fontsize=24,
        suptitle_fontsize=52, 
        x_label_fontsize=32, 
        y_label_fontsize=32
    )
    
    # Save the first plot
    quiet_output_path = os.path.join(args.output_dir, 'quiet_features.png')
    plt.savefig(quiet_output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig1)  # Close to free memory
    print(f"Saved Quiet Features plot to: {quiet_output_path}")
    
    print("Creating Loud Features plot...")
    
    # Create "Loud Features" plot
    fig2 = plot_multi_task_metrics(
        loud_configs,
        figsize=(30, 10),
        nrows=1, 
        ncols=4,
        show_grid=False,
        phase_transition_label="Phase Transition",
        random_baseline_label="Random Baseline",
        x_label="Training FLOPs",
        rect=[0.05, 0.12, 1, 0.97],
        x_loc=(0.5, 0.03),
        test_loc=(0.02, 0.4),
        bbox_to_anchor=(0.5, 0.02),
        w_pad=0,
        phase_style="red_line",
        super_title="Loud Features",
        suptitle_fontsize=72,
        x_label_fontsize=52,
        y_label_fontsize=52,
        axes_title_fontsize=42,
        axis_tick_params_labelsize=32,
        markersize=24,
        tst_fontsize=36,
        vertical_linewidth=4,
        horizontal_linewidth=8
    )
    
    # Save the second plot
    loud_output_path = os.path.join(args.output_dir, 'loud_features.png')
    plt.savefig(loud_output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig2)  # Close to free memory
    print(f"Saved Loud Features plot to: {loud_output_path}")
    
    print("All plots saved successfully!")


if __name__ == "__main__":
    main()