"""
Plot side-by-side TVP-VAR net pairwise connectedness networks.

The networks use time-averaged NPDC values. For the Diebold-Yilmaz table
orientation used here, rows are receivers and columns are transmitters, so a
positive NPDC(i, j) is drawn as j -> i.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

tmp_cache = PROJECT_ROOT / "results" / "tmp_mpl"
tmp_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(tmp_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.paths import FIGURES_DIR, TABLES_DIR


NODE_LABELS = {
    "bitcoin_conditional_volatility": "BTC",
    "ndx_conditional_volatility": "NDX",
    "nvda_conditional_volatility": "NVDA",
    "googl_conditional_volatility": "GOOGL",
    "msft_conditional_volatility": "MSFT",
}

SYSTEM_LABELS = {
    "ai_equity": "BTC-AI Equity System",
    "benchmark": "BTC-NDX System",
}

SYSTEMS = ("benchmark", "ai_equity")

EDGE_BLUE = "#0000ff"
EDGE_ORANGE = "#ff9800"


@dataclass(frozen=True)
class EdgeData:
    source: str
    target: str
    npdc: float


@dataclass(frozen=True)
class NetworkData:
    system: str
    nodes: list[str]
    edges: list[EdgeData]
    net_values: dict[str, float]
    edge_values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot side-by-side TVP-VAR connectedness networks.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TABLES_DIR / "tvpvar_connectedness",
        help="Directory containing TVP-VAR connectedness CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR / "tvpvar_connectedness",
        help="Directory for network figures.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[10, 100],
        help="Forecast horizons to plot.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum absolute average NPDC value required for an edge to be shown.",
    )
    parser.add_argument(
        "--blue-threshold",
        type=float,
        default=None,
        help="Absolute average NPDC value above which edges are drawn blue. Defaults to the 75th percentile of shown edges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for horizon in args.horizons:
        networks = [
            _load_network(args.input_dir, system, horizon, edge_threshold=args.edge_threshold)
            for system in SYSTEMS
        ]
        networks = [network for network in networks if network.nodes]
        if not networks:
            print(f"Skipping h={horizon}; no connectedness network data found.")
            continue

        blue_threshold = _blue_threshold(networks, args.blue_threshold)
        output_path = args.output_dir / f"network_side_by_side_h{horizon}.png"
        _plot_side_by_side_networks(networks, horizon, blue_threshold, output_path)
        outputs.append(output_path)

    for output in outputs:
        print(output.relative_to(PROJECT_ROOT))


def _load_network(input_dir: Path, system: str, horizon: int, *, edge_threshold: float) -> NetworkData:
    npdc_path = input_dir / f"tvpvar_connectedness_{system}_h{horizon}_npdc_h{horizon}.csv"
    net_path = input_dir / f"tvpvar_connectedness_{system}_h{horizon}_net_h{horizon}.csv"
    if not npdc_path.exists() or not net_path.exists():
        return NetworkData(system=system, nodes=[], edges=[], net_values={}, edge_values=[])

    avg_npdc = (
        pd.read_csv(npdc_path)
        .groupby(["dimension_1", "dimension_2"], as_index=False)["value"]
        .mean()
    )
    avg_net = pd.read_csv(net_path).drop(columns=["date"], errors="ignore").mean(numeric_only=True)

    nodes = list(avg_net.index)
    net_values = {str(variable): float(net_value) for variable, net_value in avg_net.items()}
    edges: list[EdgeData] = []
    edge_values: list[float] = []
    seen_pairs: set[frozenset[str]] = set()
    for _, row in avg_npdc.iterrows():
        receiver = str(row["dimension_1"])
        transmitter = str(row["dimension_2"])
        value = float(row["value"])
        if receiver == transmitter or abs(value) <= edge_threshold:
            continue
        pair = frozenset((receiver, transmitter))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        if value > 0:
            source, target = transmitter, receiver
        else:
            source, target = receiver, transmitter

        if source not in net_values or target not in net_values:
            continue
        strength = abs(value)
        edges.append(EdgeData(source=source, target=target, npdc=strength))
        edge_values.append(strength)

    return NetworkData(system=system, nodes=nodes, edges=edges, net_values=net_values, edge_values=edge_values)


def _blue_threshold(networks: list[NetworkData], configured_threshold: float | None) -> float:
    if configured_threshold is not None:
        return configured_threshold

    edge_values = [value for network in networks for value in network.edge_values]
    if not edge_values:
        return np.inf
    return float(np.percentile(edge_values, 75))


def _plot_side_by_side_networks(
    networks: list[NetworkData],
    horizon: int,
    blue_threshold: float,
    output_path: Path,
) -> None:
    max_abs_net = max((abs(value) for network in networks for value in network.net_values.values()), default=1.0)
    max_edge = max((value for network in networks for value in network.edge_values), default=1.0)

    fig, axes = plt.subplots(1, len(networks), figsize=(15.5, 7.2), constrained_layout=True)
    if len(networks) == 1:
        axes = [axes]

    for ax, network in zip(axes, networks):
        _draw_network_panel(
            ax,
            network,
            horizon=horizon,
            max_abs_net=max_abs_net,
            max_edge=max_edge,
            blue_threshold=blue_threshold,
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_network_panel(
    ax: plt.Axes,
    network: NetworkData,
    *,
    horizon: int,
    max_abs_net: float,
    max_edge: float,
    blue_threshold: float,
) -> None:
    pos = _network_positions(network.nodes)
    node_radii = {
        node: 0.09 + 0.16 * abs(network.net_values[node]) / max(max_abs_net, 1e-12)
        for node in network.nodes
    }
    for edge in sorted(network.edges, key=lambda item: item.npdc):
        source_x, source_y, target_x, target_y = _shortened_edge(
            edge.source,
            edge.target,
            pos,
            node_radii,
        )
        edge_color = EDGE_BLUE if edge.npdc > blue_threshold else EDGE_ORANGE
        edge_width = 0.9 + 3.2 * edge.npdc / max(max_edge, 1e-12)
        ax.annotate(
            "",
            xy=(target_x, target_y),
            xytext=(source_x, source_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": edge_color,
                "lw": edge_width,
                "mutation_scale": 24,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": "arc3,rad=0",
            },
            alpha=0.9,
            zorder=2,
        )

    for node in network.nodes:
        x, y = pos[node]
        net_value = network.net_values[node]
        radius = node_radii[node]
        node_color = "white" if net_value >= 0 else "black"
        circle = plt.Circle(
            (x, y),
            radius=radius,
            facecolor=node_color,
            edgecolor="black",
            linewidth=1.8,
            zorder=3,
        )
        ax.add_patch(circle)
        _draw_node_label(ax, node, x, y, radius)

    ax.set_title(
        f"{SYSTEM_LABELS.get(network.system, network.system)}, H = {horizon}",
        fontsize=17,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _network_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    if len(nodes) == 2:
        angles = np.array([np.pi, 0.0])
    else:
        angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(nodes), endpoint=False)

    return {
        node: (float(np.cos(angle)), float(np.sin(angle)))
        for node, angle in zip(nodes, angles)
    }


def _shortened_edge(
    source: str,
    target: str,
    positions: dict[str, tuple[float, float]],
    node_radii: dict[str, float],
) -> tuple[float, float, float, float]:
    source_x, source_y = positions[source]
    target_x, target_y = positions[target]
    dx = target_x - source_x
    dy = target_y - source_y
    distance = math.hypot(dx, dy)
    if distance == 0:
        return source_x, source_y, target_x, target_y

    unit_x = dx / distance
    unit_y = dy / distance
    source_offset = node_radii[source] + 0.03
    target_offset = node_radii[target] + 0.03
    return (
        source_x + unit_x * source_offset,
        source_y + unit_y * source_offset,
        target_x - unit_x * target_offset,
        target_y - unit_y * target_offset,
    )


def _draw_node_label(ax: plt.Axes, node: str, x: float, y: float, radius: float) -> None:
    label = NODE_LABELS.get(node, node)
    distance = 0.13 + radius
    angle = math.atan2(y, x)
    label_x = x + distance * math.cos(angle)
    label_y = y + distance * math.sin(angle)
    ha = "left" if label_x > 0.08 else "right" if label_x < -0.08 else "center"
    va = "bottom" if label_y > 0.08 else "top" if label_y < -0.08 else "center"
    ax.text(
        label_x,
        label_y,
        label,
        ha=ha,
        va=va,
        fontsize=11,
        fontweight="bold",
        color="black",
        zorder=4,
    )


if __name__ == "__main__":
    main()
