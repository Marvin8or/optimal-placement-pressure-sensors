# -*- coding: utf-8 -*-
"""
Result Visualization for Sensor Configuration Evaluation.

Aggregates PSO-based leak-localization results across all tested
sensor configurations and displays an interactive visualization of
localization rates with per-sensor-count breakdowns.

@author: gabri
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.widgets as mwidgets
import mplcursors


# ── Colour palette (colourblind-friendly, up to 10 sensor counts) ──────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def get_df_from_csv(csv_file_path):
    """Read a single combo result CSV."""
    return pd.read_csv(csv_file_path, index_col=0)


def unify_sensor_combination_results(num_sensors_folder_name):
    """Concatenate all individual-combo CSVs under *num_sensors_folder_name*."""
    result_dfs = []
    for path, _, files in os.walk(num_sensors_folder_name):
        for f in files:
            if f.endswith(".csv"):
                result_dfs.append(get_df_from_csv(os.path.join(path, f)))
    if not result_dfs:
        return pd.DataFrame()
    return pd.concat(result_dfs, axis=0)


def unify_all_sensor_number_results(network_evaluation_folder_name):
    """Unify results across every sensor-count sub-folder."""
    result_dfs = []
    for entry in sorted(os.listdir(network_evaluation_folder_name)):
        sub = os.path.join(network_evaluation_folder_name, entry)
        if os.path.isdir(sub):
            df = unify_sensor_combination_results(sub)
            if not df.empty:
                result_dfs.append(df)
    unified_df = pd.concat(result_dfs, axis=0).reset_index(drop=True)
    return unified_df


def _count_sensors(combo_str):
    """Extract the number of sensors from the tuple-like string."""
    return len(re.findall(r"'[^']+'", str(combo_str)))


def _short_label(combo_str):
    """Build a compact label from a sensor-combination string."""
    names = re.findall(r"'([^']+)'", str(combo_str))
    short = [n.replace("JUNCTION-", "J").replace("RESERVOIR-", "R") for n in names]
    return ", ".join(short)


def parse_inp_network(inp_path):
    """Parse an EPANET .inp file and return node coordinates and pipe edges.

    Returns
    -------
    node_coords : dict
        ``{node_id: (x, y)}`` for every node that has [COORDINATES].
    edges : list[tuple[str, str]]
        ``[(node1, node2), ...]`` from [PIPES] (+ [PUMPS] / [VALVES]).
    """
    node_coords = {}
    edges = []

    with open(inp_path, "r") as fh:
        section = None
        for raw_line in fh:
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line.upper()
                continue
            if not line or line.startswith(";"):
                continue

            parts = line.split()
            if section == "[COORDINATES]" and len(parts) >= 3:
                node_id = parts[0]
                x, y = float(parts[1]), float(parts[2])
                node_coords[node_id] = (x, y)
            elif section in ("[PIPES]", "[PUMPS]", "[VALVES]") and len(parts) >= 3:
                _link_id, node1, node2 = parts[0], parts[1], parts[2]
                edges.append((node1, node2))

    return node_coords, edges


def build_summary_dataframe(unified_df):
    """Return a per-combination summary with rate and sensor count."""
    grouped = unified_df.groupby("Sensor combination")
    records = []
    for combo_name, grp in grouped:
        n_correct = grp["Correctly located leak"].sum()
        n_total = grp.shape[0]
        rate = 100.0 * n_correct / n_total
        records.append({
            "combo": combo_name,
            "label": _short_label(combo_name),
            "num_sensors": _count_sensors(combo_name),
            "rate": rate,
            "n_scenarios": n_total,
        })
    summary = pd.DataFrame(records)
    summary.sort_values("rate", ascending=True, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    summary["rank"] = range(1, len(summary) + 1)
    return summary


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(summary, node_coords=None, edges=None):
    """Create a multi-panel figure with network map, scatter, box-plot, and filter.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of ``build_summary_dataframe``.
    node_coords : dict or None
        ``{node_id: (x, y)}`` from ``parse_inp_network``.
    edges : list[tuple[str,str]] or None
        Pipe/pump/valve edges from ``parse_inp_network``.
    """

    sensor_counts = sorted(summary["num_sensors"].unique())
    color_map = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sensor_counts)}

    # Track which sensor counts are currently visible
    visible = {n: True for n in sensor_counts}

    has_network = node_coords is not None and edges is not None

    # ── Figure layout ───────────────────────────────────────────────────
    if has_network:
        fig = plt.figure(figsize=(18, 12), facecolor="#FAFAFA")
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[0.8, 3, 1],
            height_ratios=[1, 1],
            wspace=0.08, hspace=0.28,
            left=0.04, right=0.97, top=0.95, bottom=0.05,
        )
        ax_check = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[0, 1])
        ax_box = fig.add_subplot(gs[0, 2])
        ax_net = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(17, 7), facecolor="#FAFAFA")
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[0.8, 3, 1],
            wspace=0.08,
            left=0.04, right=0.97, top=0.92, bottom=0.08,
        )
        ax_check = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[0, 1])
        ax_box = fig.add_subplot(gs[0, 2])
        ax_net = None

    for ax in (ax_scatter, ax_box):
        ax.set_facecolor("#FAFAFA")

    # ── Draw static network map ────────────────────────────────────────
    if has_network and ax_net is not None:
        ax_net.set_facecolor("#FAFAFA")
        # Draw edges (pipes)
        for n1, n2 in edges:
            if n1 in node_coords and n2 in node_coords:
                x1, y1 = node_coords[n1]
                x2, y2 = node_coords[n2]
                ax_net.plot([x1, x2], [y1, y2], color="#CCCCCC", lw=0.6, zorder=1)
        # Draw all nodes (gray)
        all_x = [c[0] for c in node_coords.values()]
        all_y = [c[1] for c in node_coords.values()]
        ax_net.scatter(all_x, all_y, s=14, c="#BBBBBB", edgecolors="white",
                       linewidths=0.3, zorder=2)
        ax_net.set_aspect("equal", adjustable="datalim")
        ax_net.set_title("Network Map  (hover a point above to highlight sensors)",
                         fontsize=11, weight="bold")
        ax_net.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_net.spines[["top", "right", "bottom", "left"]].set_visible(False)

    # ── Helper: (re-)draw both panels for currently visible groups ──────
    # Keep mutable references so the callback can swap them out
    state = {
        "scatter_artists": {},   # n -> PathCollection
        "mean_line": None,
        "median_line": None,
        "filtered_summary": summary.copy(),
        "cursor": None,
        "highlight_artist": None,  # red sensor dots on network map
        "highlight_labels": [],    # node-id labels on network map
    }

    def _redraw():
        # ── Scatter panel ───────────────────────────────────────────
        # Remove old artists
        for art in state["scatter_artists"].values():
            art.remove()
        state["scatter_artists"].clear()
        if state["mean_line"] is not None:
            state["mean_line"].remove()
            state["mean_line"] = None
        if state["median_line"] is not None:
            state["median_line"].remove()
            state["median_line"] = None

        # Filter & re-rank
        active = [n for n in sensor_counts if visible[n]]
        if not active:
            state["filtered_summary"] = summary.iloc[0:0].copy()
            ax_scatter.set_xlim(0, 1)
            ax_scatter.legend(fontsize=8, loc="upper left", framealpha=0.9)
            # Clear box plot
            ax_box.cla()
            ax_box.set_facecolor("#FAFAFA")
            ax_box.set_xlabel("Number of sensors", fontsize=11)
            ax_box.set_title("Distribution", fontsize=13, weight="bold")
            fig.canvas.draw_idle()
            return

        filtered = summary[summary["num_sensors"].isin(active)].copy()
        filtered.sort_values("rate", ascending=True, inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        filtered["rank"] = range(1, len(filtered) + 1)
        state["filtered_summary"] = filtered

        for n in active:
            mask = filtered["num_sensors"] == n
            art = ax_scatter.scatter(
                filtered.loc[mask, "rank"],
                filtered.loc[mask, "rate"],
                c=color_map[n],
                label=f"{n} sensors",
                s=36,
                alpha=0.70,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            state["scatter_artists"][n] = art

        # Mean & median
        mean_rate = filtered["rate"].mean()
        median_rate = filtered["rate"].median()
        state["mean_line"] = ax_scatter.axhline(
            mean_rate, color="#888888", ls="--", lw=1, zorder=2,
            label=f"Mean  {mean_rate:.1f}%",
        )
        state["median_line"] = ax_scatter.axhline(
            median_rate, color="#AAAAAA", ls=":", lw=1, zorder=2,
            label=f"Median  {median_rate:.1f}%",
        )

        ax_scatter.set_xlim(0, len(filtered) + 1)
        ax_scatter.legend(fontsize=8, loc="upper left", framealpha=0.9)

        # ── Box-plot panel ──────────────────────────────────────────
        ax_box.cla()
        ax_box.set_facecolor("#FAFAFA")
        box_data = [filtered.loc[filtered["num_sensors"] == n, "rate"].values
                    for n in active]
        if box_data:
            bp = ax_box.boxplot(
                box_data,
                patch_artist=True,
                labels=[str(n) for n in active],
                widths=0.55,
                showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
                medianprops=dict(color="#333333", lw=1.5),
                whiskerprops=dict(color="#888888"),
                capprops=dict(color="#888888"),
            )
            for patch, n in zip(bp["boxes"], active):
                patch.set_facecolor(color_map[n])
                patch.set_alpha(0.65)

        ax_box.set_xlabel("Number of sensors", fontsize=11)
        ax_box.set_title("Distribution", fontsize=13, weight="bold")
        ax_box.set_ylim(ax_scatter.get_ylim())
        ax_box.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax_box.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=1)
        ax_box.tick_params(axis="y", labelleft=False, labelsize=9)
        ax_box.tick_params(axis="x", labelsize=9)
        ax_box.spines[["top", "right"]].set_visible(False)

        # Refresh hover cursor
        if state["cursor"] is not None:
            state["cursor"].remove()
        state["cursor"] = mplcursors.cursor(
            list(state["scatter_artists"].values()), hover=True,
        )

        @state["cursor"].connect("add")
        def _on_hover(sel):
            x_val = sel.target[0]
            filt = state["filtered_summary"]
            row = filt[filt["rank"] == round(x_val)]
            if row.empty:
                sel.annotation.set_text("")
                return
            row = row.iloc[0]
            sel.annotation.set_text(
                f"Nodes IDs: {row['label']}\n"
                f"Localization Rate: {row['rate']:.1f}%\n"
                f"Sensors: {row['num_sensors']}  |  "
                f"Evaluated Scenarios: {row['n_scenarios']}"
            )
            sel.annotation.get_bbox_patch().set(
                fc="#FFFFFFDD", ec="#CCCCCC", boxstyle="round,pad=0.4",
            )

            # ── Highlight sensors on the network map ────────────────
            if has_network and ax_net is not None:
                # Remove previous highlights
                if state["highlight_artist"] is not None:
                    state["highlight_artist"].remove()
                    state["highlight_artist"] = None
                for lbl in state["highlight_labels"]:
                    lbl.remove()
                state["highlight_labels"].clear()

                sensor_names = re.findall(r"'([^']+)'", str(row["combo"]))
                sx = [node_coords[n][0] for n in sensor_names if n in node_coords]
                sy = [node_coords[n][1] for n in sensor_names if n in node_coords]
                if sx:
                    state["highlight_artist"] = ax_net.scatter(
                        sx, sy, s=80, c="red", edgecolors="darkred",
                        linewidths=0.8, zorder=4,
                    )
                    for n in sensor_names:
                        if n in node_coords:
                            x, y = node_coords[n]
                            short = n.replace("JUNCTION-", "J").replace("RESERVOIR-", "R")
                            lbl = ax_net.annotate(
                                short, (x, y),
                                textcoords="offset points", xytext=(6, 6),
                                fontsize=7, color="red", weight="bold", zorder=5,
                            )
                            state["highlight_labels"].append(lbl)
                    ax_net.set_title(
                        f"Network Map  —  {row['label']}  "
                        f"({row['rate']:.1f}%)",
                        fontsize=11, weight="bold",
                    )

        fig.canvas.draw_idle()

    # ── Static scatter/box axes styling (set once) ──────────────────────
    ax_scatter.set_ylim(-2, 105)
    ax_scatter.set_xlabel("Configuration (ranked by localization rate)", fontsize=11)
    ax_scatter.set_ylabel("Localization rate (%)", fontsize=11)
    ax_scatter.set_title("Sensor-Configuration Performance", fontsize=13, weight="bold")
    ax_scatter.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_scatter.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=1)
    ax_scatter.tick_params(axis="x", labelsize=9)
    ax_scatter.tick_params(axis="y", labelsize=9)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    # ── CheckButtons widget (filter by sensor count) ────────────────────
    ax_check.set_facecolor("#FAFAFA")
    ax_check.set_title("Filter", fontsize=11, weight="bold")
    labels = [f"{n} sensors" for n in sensor_counts]
    initial_states = [True] * len(sensor_counts)

    check = mwidgets.CheckButtons(
        ax_check, labels, initial_states,
    )
    # Style the check-button labels & boxes to match the palette
    for i, n in enumerate(sensor_counts):
        check.labels[i].set_fontsize(9)
        check.labels[i].set_color(color_map[n])
        for rect in check.rectangles if hasattr(check, 'rectangles') else []:
            pass  # styling handled by matplotlib internals

    def _on_check(label):
        idx = labels.index(label)
        n = sensor_counts[idx]
        visible[n] = not visible[n]
        _redraw()

    check.on_clicked(_on_check)

    # Initial draw
    _redraw()
    plt.show()

    # prevent garbage-collection of the widget while the window is open
    return check


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INP_FILE = os.path.join(os.getcwd(), "networks", "Net1.inp")
    rootfolder = os.path.join(
        os.getcwd(), "Net1_evaluation", "sensor_configurations"
    )

    print("Loading results …")
    unified_df = unify_all_sensor_number_results(rootfolder)
    summary = build_summary_dataframe(unified_df)

    n_configs = len(summary)
    n_sensor_groups = summary["num_sensors"].nunique()
    print(f"  {n_configs} configurations across "
          f"{n_sensor_groups} sensor-count group(s).")
    print(f"  Best rate: {summary['rate'].max():.1f}%  |  "
          f"Mean: {summary['rate'].mean():.1f}%  |  "
          f"Worst: {summary['rate'].min():.1f}%")

    best = summary.loc[summary["rate"].idxmax()]
    print(f"  Best config: {best['label']} ({best['rate']:.1f}%)\n")

    # Parse network topology for the map panel
    node_coords, edges = None, None
    if os.path.isfile(INP_FILE):
        print(f"Loading network from {INP_FILE} …")
        node_coords, edges = parse_inp_network(INP_FILE)
        print(f"  {len(node_coords)} nodes, {len(edges)} links\n")
    else:
        print(f"  Network file not found ({INP_FILE}), skipping map panel.\n")

    _check_widget = plot_results(summary, node_coords, edges)
