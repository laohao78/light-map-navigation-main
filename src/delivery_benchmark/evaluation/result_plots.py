from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_eval1_summary(records: Sequence[dict], summary: dict, output_path: str | Path) -> None:
    if not records:
        raise ValueError('No records available for plotting')

    output_path = _prepare_output_path(output_path)
    buildings = [record['building_name'] for record in records]
    baseline = np.asarray([record['baseline_prefix_3_cost'] for record in records], dtype=float)
    optimized = np.asarray([record['optimized_prefix_3_cost'] for record in records], dtype=float)
    improvements = baseline - optimized

    x_positions = np.arange(len(buildings))
    width = 0.38

    fig, (ax_cost, ax_gain) = plt.subplots(2, 1, figsize=(max(10, len(buildings) * 0.75), 8), sharex=True)

    ax_cost.bar(x_positions - width / 2, baseline, width=width, label='Baseline', color='#7F8C8D')
    ax_cost.bar(x_positions + width / 2, optimized, width=width, label='Optimized', color='#2E86AB')
    ax_cost.set_ylabel('Prefix cost')
    ax_cost.set_title('Eval1: Prefix-3 Cost Comparison')
    ax_cost.legend(loc='upper right')

    ax_gain.bar(x_positions, improvements, color=['#16A085' if value >= 0 else '#C0392B' for value in improvements])
    ax_gain.axhline(0.0, color='black', linewidth=1)
    ax_gain.set_ylabel('Improvement')
    ax_gain.set_xlabel('Building')

    ax_gain.set_xticks(x_positions)
    ax_gain.set_xticklabels(buildings, rotation=45, ha='right')

    fig.text(
        0.01,
        0.01,
        f"Mean baseline={summary.get('baseline_prefix_3_mean', 0.0):.2f}, "
        f"optimized={summary.get('optimized_prefix_3_mean', 0.0):.2f}, "
        f"mean gain={summary.get('mean_improvement', 0.0):.2f}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_eval2_summary(records: Sequence[dict], summary: dict, output_path: str | Path) -> None:
    if not records:
        raise ValueError('No records available for plotting')

    output_path = _prepare_output_path(output_path)
    buildings = [record['building_name'] for record in records]
    baseline_target = np.asarray([record['baseline_target_cost'] for record in records], dtype=float)
    optimized_target = np.asarray([record['optimized_target_cost'] for record in records], dtype=float)
    baseline_full = np.asarray([record['baseline_full_cost'] for record in records], dtype=float)
    optimized_full = np.asarray([record['optimized_full_cost'] for record in records], dtype=float)
    target_improvement = baseline_target - optimized_target
    full_improvement = baseline_full - optimized_full

    x_positions = np.arange(len(buildings))
    width = 0.38

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(buildings) * 0.75), 11), sharex=True)

    axes[0].bar(x_positions - width / 2, baseline_target, width=width, label='Baseline target', color='#7F8C8D')
    axes[0].bar(x_positions + width / 2, optimized_target, width=width, label='Optimized target', color='#2E86AB')
    axes[0].set_ylabel('Target cost')
    axes[0].set_title('Eval2: Target and Full Path Cost Comparison')
    axes[0].legend(loc='upper right')

    axes[1].bar(x_positions - width / 2, baseline_full, width=width, label='Baseline full', color='#95A5A6')
    axes[1].bar(x_positions + width / 2, optimized_full, width=width, label='Optimized full', color='#1ABC9C')
    axes[1].set_ylabel('Full cost')
    axes[1].legend(loc='upper right')

    axes[2].bar(x_positions - width / 2, target_improvement, width=width, label='Target gain', color='#16A085')
    axes[2].bar(x_positions + width / 2, full_improvement, width=width, label='Full gain', color='#D35400')
    axes[2].axhline(0.0, color='black', linewidth=1)
    axes[2].set_ylabel('Improvement')
    axes[2].set_xlabel('Building')
    axes[2].legend(loc='upper right')

    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(buildings, rotation=45, ha='right')

    fig.text(
        0.01,
        0.01,
        f"Mean target gain={summary.get('mean_improvement', 0.0):.2f}, "
        f"mean full gain={summary.get('mean_full_cost_improvement', 0.0):.2f}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_eval3_summary(summary: dict, output_path: str | Path) -> None:
    output_path = _prepare_output_path(output_path)
    variant_names = [name for name in summary.keys() if name != 'experiment']
    if not variant_names:
        raise ValueError('No variant summary available for plotting')

    target_costs = [summary[name].get('mean_target_cost', 0.0) for name in variant_names]
    full_costs = [summary[name].get('mean_full_cost', 0.0) for name in variant_names]
    target_steps = [summary[name].get('mean_target_steps', 0.0) for name in variant_names]
    candidate_counts = [summary[name].get('mean_candidate_count', 0.0) for name in variant_names]

    x_positions = np.arange(len(variant_names))

    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(variant_names) * 1.6), 9))
    axes = axes.flatten()

    metric_specs = [
        ('Mean target cost', target_costs, '#2E86AB'),
        ('Mean full cost', full_costs, '#1ABC9C'),
        ('Mean target steps', target_steps, '#D35400'),
        ('Mean candidate count', candidate_counts, '#7F8C8D'),
    ]

    for axis, (title, values, color) in zip(axes, metric_specs):
        axis.bar(x_positions, values, color=color)
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(variant_names, rotation=35, ha='right')
        axis.grid(axis='y', alpha=0.25)

    fig.suptitle('Eval3: Ablation Study Summary', fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)