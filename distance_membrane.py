from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# -----------------------------
# Settings
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_files"
GRAPH_DIR = BASE_DIR / "graphs"
OUTPUT_DIR = BASE_DIR / "output_files"

# Your file header says Time (ps), so this should be True.
CONVERT_PS_TO_NS = True

EQUILIBRATION_CUTOFF_NS = 0

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    #"axes.titlesize": 25,
    #"axes.labelsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 20,
})


# -----------------------------
# Custom colors
# -----------------------------
light_pink = "#ff98cb"
pink = "#ff69b3"
dark_pink = "#ec3499"
darkest_pink = "#cb1378"
light_blue = "#65a9ed"
blue = "#65a9ed"
dark_blue = "#1546c7"
darkest_blue = "#000081"

VARIANT_COLORS = {
    "wt": dark_blue,
    "v8a": dark_pink,
}

VARIANT_FILL_COLORS = {
    "wt": light_blue,
    "v8a": light_pink,
}


# -----------------------------
# Reading and parsing
# -----------------------------
def make_dirs() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename(filename: str) -> dict:
    """
    Expected filenames:
        mem_mon_wt_repl1_membrane_mbd_xyzdist.xvg
        mem_mon_v8a_repl3_membrane_mbd_xyzdist.xvg
        mem_dim_wt_repl1_membrane_mbd_xyzdist.xvg
        mem_dim_v8a_repl3_membrane_mbd_xyzdist.xvg

    Also accepts the small typo:
        mem_mon_wt_repl2membrane_mbd_xyzdist.xvg
    """
    stem = filename.replace("_xyzdist.xvg", "")

    pattern = re.compile(
        r"^(?P<environment>mem)_"
        r"(?P<oligomer>mon|dim)_"
        r"(?P<variant>wt|v8a)_"
        r"(?P<replicate>repl\d+)_?"
        r"membrane_mbd$"
    )

    match = pattern.match(stem)

    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    data = match.groupdict()
    data["system"] = f"{data['environment']}_{data['oligomer']}"
    data["comparison_group"] = data["system"]
    data["condition"] = f"{data['system']}_{data['variant']}"

    return data


def read_xvg(filepath: Path) -> pd.DataFrame:
    """
    Reads a GROMACS gmx distance -oxyz .xvg file.

    Columns:
        column 1 = time
        column 2 = x-distance
        column 3 = y-distance
        column 4 = z-distance

    This script uses only the z-distance.
    """
    rows = []

    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#") or line.startswith("@"):
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                time = float(parts[0])
                x_distance = float(parts[1])
                y_distance = float(parts[2])
                z_distance = float(parts[3])

                rows.append((time, x_distance, y_distance, z_distance))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric XYZ-distance data found in {filepath}")

    df = pd.DataFrame(
        rows,
        columns=[
            "time_raw",
            "x_distance_nm",
            "y_distance_nm",
            "z_distance_nm",
        ],
    )

    df["time_ns"] = df["time_raw"] / 1000 if CONVERT_PS_TO_NS else df["time_raw"]

    return df


def load_all_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_membrane_mbd_xyzdist.xvg"))

    if not files:
        raise FileNotFoundError(
            f"No *_membrane_mbd_xyzdist.xvg files found in {INPUT_DIR}"
        )

    all_frames = []

    for filepath in files:
        meta = parse_filename(filepath.name)
        df = read_xvg(filepath)

        for key, value in meta.items():
            df[key] = value

        df["source_file"] = filepath.name
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    combined["replicate_number"] = (
        combined["replicate"]
        .str.replace("repl", "", regex=False)
        .astype(int)
    )

    combined = combined.sort_values(
        ["system", "variant", "replicate_number", "time_ns"]
    ).reset_index(drop=True)

    return combined


# -----------------------------
# Summaries
# -----------------------------
def calculate_replicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = df[df["time_ns"] >= EQUILIBRATION_CUTOFF_NS].copy()

    summary = (
        stats_df.groupby(
            [
                "system",
                "environment",
                "oligomer",
                "comparison_group",
                "variant",
                "replicate",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            n_points=("z_distance_nm", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_z_distance_nm=("z_distance_nm", "mean"),
            median_z_distance_nm=("z_distance_nm", "median"),
            std_z_distance_nm=("z_distance_nm", "std"),
            min_z_distance_nm=("z_distance_nm", "min"),
            max_z_distance_nm=("z_distance_nm", "max"),
        )
        .sort_values(["system", "variant", "replicate"])
    )

    return summary


def calculate_wt_vs_v8a_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    condition_summary = (
        replicate_summary.groupby(
            ["system", "comparison_group", "variant"],
            as_index=False,
        )
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_z_distance_nm=("mean_z_distance_nm", "mean"),
            sd_mean_z_distance_nm=("mean_z_distance_nm", "std"),
            mean_max_z_distance_nm=("max_z_distance_nm", "mean"),
            sd_max_z_distance_nm=("max_z_distance_nm", "std"),
            mean_min_z_distance_nm=("min_z_distance_nm", "mean"),
            sd_min_z_distance_nm=("min_z_distance_nm", "std"),
        )
        .sort_values(["system", "variant"])
    )

    return condition_summary


def calculate_wt_vs_v8a_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = replicate_summary.pivot_table(
        index=["system", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_z_distance_nm", "max_z_distance_nm", "min_z_distance_nm"],
    )

    rows = []

    for index, row in pivot.iterrows():
        system, comparison_group, replicate = index

        try:
            rows.append({
                "system": system,
                "comparison_group": comparison_group,
                "replicate": replicate,
                "mean_z_distance_difference_v8a_minus_wt_nm":
                    row[("mean_z_distance_nm", "v8a")]
                    - row[("mean_z_distance_nm", "wt")],
                "max_z_distance_difference_v8a_minus_wt_nm":
                    row[("max_z_distance_nm", "v8a")]
                    - row[("max_z_distance_nm", "wt")],
                "min_z_distance_difference_v8a_minus_wt_nm":
                    row[("min_z_distance_nm", "v8a")]
                    - row[("min_z_distance_nm", "wt")],
            })
        except KeyError:
            continue

    return pd.DataFrame(rows)


def calculate_t_tests(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        wt = sub[sub["variant"] == "wt"]
        v8a = sub[sub["variant"] == "v8a"]

        if wt.empty or v8a.empty:
            continue

        mean_t, mean_p = stats.ttest_ind(
            wt["mean_z_distance_nm"],
            v8a["mean_z_distance_nm"],
            equal_var=False,
        )

        rows.append({
            "comparison_group": comparison_group,
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),
            "wt_mean_z_distance_nm": wt["mean_z_distance_nm"].mean(),
            "wt_sd_z_distance_nm": wt["mean_z_distance_nm"].std(),
            "v8a_mean_z_distance_nm": v8a["mean_z_distance_nm"].mean(),
            "v8a_sd_z_distance_nm": v8a["mean_z_distance_nm"].std(),
            "mean_z_distance_t_statistic": mean_t,
            "mean_z_distance_p_value": mean_p,
            "mean_z_distance_significant_p_lt_0_05": mean_p < 0.05,
        })

    return pd.DataFrame(rows)


def save_tables(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
) -> None:
    df.to_csv(
        OUTPUT_DIR / "all_membrane_mbd_z_distance_long_format.csv",
        index=False,
    )

    replicate_summary.to_csv(
        OUTPUT_DIR / "membrane_mbd_z_distance_summary_by_replicate.csv",
        index=False,
    )

    condition_summary.to_csv(
        OUTPUT_DIR / "wt_vs_v8a_membrane_mbd_z_distance_summary_by_group.csv",
        index=False,
    )

    differences.to_csv(
        OUTPUT_DIR / "wt_vs_v8a_membrane_mbd_z_distance_differences_by_replicate.csv",
        index=False,
    )

    t_tests.to_csv(
        OUTPUT_DIR / "wt_vs_v8a_membrane_mbd_z_distance_t_tests.csv",
        index=False,
    )


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


TITLE_MAP = {
    "mem_mon": "Z-distance Between Membrane Binding Domain and Membrane Middle in WT and V8A Monomers",
    "mem_dim": "Z-distance Between Membrane Binding Domain and Membrane Middle in WT and V8A Dimers",
}


def pretty_title(text: str) -> str:
    return TITLE_MAP.get(text, text.replace("_", " ").upper())


def calculate_wt_reference_means(df: pd.DataFrame) -> pd.Series:
    """
    Calculates one WT reference mean per comparison group.

    Example:
        mem_dim -> mean z-distance of all WT mem_dim data
        mem_mon -> mean z-distance of all WT mem_mon data
    """
    stats_df = df[df["time_ns"] >= EQUILIBRATION_CUTOFF_NS].copy()

    wt_reference_means = (
        stats_df[stats_df["variant"] == "wt"]
        .groupby("comparison_group")["z_distance_nm"]
        .mean()
    )

    return wt_reference_means


def plot_wt_vs_v8a_replicate_pairs(df: pd.DataFrame) -> None:

    for (comparison_group, replicate), sub in df.groupby(
        ["comparison_group", "replicate"]
    ):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = (
                sub[sub["variant"] == variant]
                .sort_values("time_ns")
            )

            plt.plot(
                variant_df["time_ns"],
                variant_df["z_distance_nm"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.axhline(
            0,
            color="black",
            linewidth=1,
            linestyle="--",
        )

        plt.xlabel("Time (ns)")
        plt.ylabel("COM z-distance (nm)")
        plt.xlim(0, 300)
        #plt.ylim(1.5, 3.5)

        legend = plt.legend(
            title=replicate.upper(),
            loc="upper left",
        )

        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"01_{comparison_group}_{replicate}_wt_vs_v8a_z_distance.png"
        )

def plot_wt_vs_v8a_mean_sd(df: pd.DataFrame) -> None:
    plot_df = df.copy()

    # Exclude problematic WT dimer replicate from the 02 mean ± SD plot
    plot_df = plot_df[
        ~(
            (plot_df["comparison_group"] == "mem_dim")
            & (plot_df["variant"] == "wt")
            & (plot_df["replicate"] == "repl1")
        )
    ].copy()

    wt_reference_means = (
        plot_df[
            (plot_df["time_ns"] >= EQUILIBRATION_CUTOFF_NS)
            & (plot_df["variant"] == "wt")
        ]
        .groupby("comparison_group")["z_distance_nm"]
        .mean()
    )

    grouped = (
        plot_df.groupby(["comparison_group", "variant", "time_ns"], as_index=False)
        .agg(
            mean_z_distance_nm=("z_distance_nm", "mean"),
            sd_z_distance_nm=("z_distance_nm", "std"),
        )
    )

    grouped["wt_reference_mean_nm"] = grouped["comparison_group"].map(
        wt_reference_means
    )

    grouped["mean_z_distance_relative_to_wt_mean_nm"] = (
        grouped["mean_z_distance_nm"] - grouped["wt_reference_mean_nm"]
    )

    grouped["lower"] = (
        grouped["mean_z_distance_relative_to_wt_mean_nm"]
        - grouped["sd_z_distance_nm"].fillna(0)
    )

    grouped["upper"] = (
        grouped["mean_z_distance_relative_to_wt_mean_nm"]
        + grouped["sd_z_distance_nm"].fillna(0)
    )

    for comparison_group, sub in grouped.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["mean_z_distance_relative_to_wt_mean_nm"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

            plt.fill_between(
                variant_df["time_ns"],
                variant_df["lower"],
                variant_df["upper"],
                color=VARIANT_FILL_COLORS[variant],
                alpha=0.20,
            )

        plt.axhline(0, color="black", linewidth=1, linestyle="--")

        plt.xlabel("Time (ns)")
        plt.ylabel("Z-distance relative to WT mean (nm)")
        plt.xlim(0, 300)
        plt.ylim(-1, 1)

        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"02_{comparison_group}_wt_vs_v8a_mean_sd_z_distance_normalized_to_wt_mean.png"
        )

def plot_wt_vs_v8a_mean_z_distance_bars(
    replicate_summary: pd.DataFrame,
) -> None:
    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="mean_z_distance_nm",
            errorbar="sd",
            capsize=0.15,
            order=["wt", "v8a"],
            palette=[dark_blue, dark_pink],
            err_kws={
                "color": "black",
                "linewidth": 2,
            },
        )

        sns.stripplot(
            data=sub,
            x="variant",
            y="mean_z_distance_nm",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        plt.axhline(0, color="black", linewidth=1, linestyle="--")

        plt.xlabel("Variant")
        plt.ylabel(
            f"Mean COM z-distance (nm)"
        )

        save_plot(f"03_{comparison_group}_mean_z_distance_bar.png")


def plot_all_mean_z_distance_comparisons_one_figure(
    replicate_summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(12, 10))

    comparison_order = [
        "mem_mon",
        "mem_dim",
    ]

    plot_df = replicate_summary.copy()
    plot_df["comparison_group"] = pd.Categorical(
        plot_df["comparison_group"],
        categories=comparison_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values("comparison_group")

    sns.barplot(
        data=plot_df,
        x="comparison_group",
        y="mean_z_distance_nm",
        hue="variant",
        errorbar="sd",
        capsize=0.15,
        hue_order=["wt", "v8a"],
        palette={"wt": dark_blue, "v8a": dark_pink},
        err_kws={
            "color": "black",
            "linewidth": 2,
        },
    )

    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(
        handles=handles,
        labels=["WT", "V8A"],
        loc="upper left",
        title=None
    )

    plt.xlabel("Comparison group")
    plt.ylabel(
        f"Mean COM z-distance (nm)"
    )

    custom_labels = [
        "Membrane Monomer",
        "Membrane Dimer",
    ]

    plt.xticks(
        ticks=range(len(custom_labels)),
        labels=custom_labels,
        rotation=25,
        ha="right",
    )

    save_plot("04_all_groups_wt_vs_v8a_mean_z_distance.png")


def make_all_plots(df: pd.DataFrame, replicate_summary: pd.DataFrame) -> None:
    plot_wt_vs_v8a_replicate_pairs(df)
    plot_wt_vs_v8a_mean_sd(df)
    plot_wt_vs_v8a_mean_z_distance_bars(replicate_summary)
    plot_all_mean_z_distance_comparisons_one_figure(replicate_summary)


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    make_dirs()

    print(f"Reading XVG files from: {INPUT_DIR}")

    df = load_all_data()

    print(f"Loaded {df['source_file'].nunique()} files")
    print(f"Total data points: {len(df):,}")

    replicate_summary = calculate_replicate_summary(df)
    condition_summary = calculate_wt_vs_v8a_summary(replicate_summary)
    differences = calculate_wt_vs_v8a_differences(replicate_summary)
    t_tests = calculate_t_tests(replicate_summary)

    save_tables(
        df,
        replicate_summary,
        condition_summary,
        differences,
        t_tests,
    )

    make_all_plots(df, replicate_summary)

    print("\nDone!")


if __name__ == "__main__":
    main()