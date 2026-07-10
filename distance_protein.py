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

CONVERT_PS_TO_NS = True
EQUILIBRATION_CUTOFF_NS = 50

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
    "axes.titlesize": 35,
    "axes.labelsize": 35,
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
    Example:
        mem_dim_v8a_repl1_helix_a_cc_a_comdist.xvg
        mem_dim_v8a_repl1_helix_b_cc_b_comdist.xvg
    """
    stem = filename.replace("_comdist.xvg", "")

    pattern = re.compile(
        r"^(?P<environment>mem|pro)_"
        r"(?P<oligomer>mon|dim)_"
        r"(?P<variant>wt|v8a)_"
        r"(?P<replicate>repl\d+)_"
        r"helix_(?P<helix>a|b)_cc_(?P<selection>a|b)$"
    )

    match = pattern.match(stem)

    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    data = match.groupdict()
    data["system"] = f"{data['environment']}_{data['oligomer']}"
    data["comparison_group"] = f"{data['system']}_{data['selection']}"
    data["condition"] = f"{data['system']}_{data['variant']}_{data['selection']}"

    return data


def read_xvg(filepath: Path) -> pd.DataFrame:
    rows = []

    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#") or line.startswith("@"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                time = float(parts[0])
                com_distance = float(parts[1])
                rows.append((time, com_distance))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric COM-distance data found in {filepath}")

    df = pd.DataFrame(rows, columns=["time_raw", "com_distance_nm"])
    df["time_ns"] = df["time_raw"] / 1000 if CONVERT_PS_TO_NS else df["time_raw"]

    return df


def load_all_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_comdist.xvg"))

    if not files:
        raise FileNotFoundError(f"No *_comdist.xvg files found in {INPUT_DIR}")

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
        ["system", "selection", "variant", "replicate_number", "time_ns"]
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
                "selection",
                "comparison_group",
                "variant",
                "replicate",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            n_points=("com_distance_nm", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_com_distance_nm=("com_distance_nm", "mean"),
            median_com_distance_nm=("com_distance_nm", "median"),
            std_com_distance_nm=("com_distance_nm", "std"),
            min_com_distance_nm=("com_distance_nm", "min"),
            max_com_distance_nm=("com_distance_nm", "max"),
        )
        .sort_values(["system", "selection", "variant", "replicate"])
    )

    return summary


def calculate_wt_vs_v8a_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    condition_summary = (
        replicate_summary.groupby(
            ["system", "selection", "comparison_group", "variant"],
            as_index=False,
        )
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_com_distance_nm=("mean_com_distance_nm", "mean"),
            sd_mean_com_distance_nm=("mean_com_distance_nm", "std"),
            mean_max_com_distance_nm=("max_com_distance_nm", "mean"),
            sd_max_com_distance_nm=("max_com_distance_nm", "std"),
        )
        .sort_values(["system", "selection", "variant"])
    )

    return condition_summary


def calculate_wt_vs_v8a_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = replicate_summary.pivot_table(
        index=["system", "selection", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_com_distance_nm", "max_com_distance_nm"],
    )

    rows = []

    for index, row in pivot.iterrows():
        system, selection, comparison_group, replicate = index

        try:
            rows.append({
                "system": system,
                "selection": selection,
                "comparison_group": comparison_group,
                "replicate": replicate,
                "mean_com_distance_difference_v8a_minus_wt_nm":
                    row[("mean_com_distance_nm", "v8a")]
                    - row[("mean_com_distance_nm", "wt")],
                "max_com_distance_difference_v8a_minus_wt_nm":
                    row[("max_com_distance_nm", "v8a")]
                    - row[("max_com_distance_nm", "wt")],
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
            wt["mean_com_distance_nm"],
            v8a["mean_com_distance_nm"],
            equal_var=False,
        )

        rows.append({
            "comparison_group": comparison_group,
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),
            "wt_mean_com_distance_nm": wt["mean_com_distance_nm"].mean(),
            "wt_sd_com_distance_nm": wt["mean_com_distance_nm"].std(),
            "v8a_mean_com_distance_nm": v8a["mean_com_distance_nm"].mean(),
            "v8a_sd_com_distance_nm": v8a["mean_com_distance_nm"].std(),
            "mean_com_distance_t_statistic": mean_t,
            "mean_com_distance_p_value": mean_p,
            "mean_com_distance_significant_p_lt_0_05": mean_p < 0.05,
        })

    return pd.DataFrame(rows)


def save_tables(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
) -> None:
    df.to_csv(OUTPUT_DIR / "all_com_distance_long_format.csv", index=False)
    replicate_summary.to_csv(OUTPUT_DIR / "com_distance_summary_by_replicate.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "wt_vs_v8a_com_distance_summary_by_group.csv", index=False)
    differences.to_csv(OUTPUT_DIR / "wt_vs_v8a_com_distance_differences_by_replicate.csv", index=False)
    t_tests.to_csv(OUTPUT_DIR / "wt_vs_v8a_com_distance_t_tests.csv", index=False)


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


TITLE_MAP = {
    "pro_mon_a": "COM Distance of WT and V8A DivIVA Monomers",
    "pro_dim_a": "COM Distance of WT and V8A DivIVA Dimers Chain A",
    "pro_dim_b": "COM Distance of WT and V8A DivIVA Dimers Chain B",
    "mem_mon_a": "COM Distance of Membrane-Associated WT and V8A DivIVA Monomers",
    "mem_dim_a": "COM Distance of Membrane-Associated WT and V8A DivIVA Dimers Chain A",
    "mem_dim_b": "COM Distance of Membrane-Associated WT and V8A DivIVA Dimers Chain B",
}


def pretty_title(text: str) -> str:
    return TITLE_MAP.get(text, text.replace("_", " ").upper())


def plot_wt_vs_v8a_replicate_pairs(df: pd.DataFrame) -> None:
    for (comparison_group, replicate), sub in df.groupby(["comparison_group", "replicate"]):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["com_distance_nm"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel("COM Distance (nm)")
        plt.xlim(0, 300)
        plt.ylim(0.6,2.2)

        legend = plt.legend(title=replicate.upper(), loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"01_{comparison_group}_{replicate}_wt_vs_v8a_com_distance.png")


def plot_wt_vs_v8a_mean_sd(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["comparison_group", "variant", "time_ns"], as_index=False)
        .agg(
            mean_com_distance_nm=("com_distance_nm", "mean"),
            sd_com_distance_nm=("com_distance_nm", "std"),
        )
    )

    grouped["lower"] = (
        grouped["mean_com_distance_nm"]
        - grouped["sd_com_distance_nm"].fillna(0)
    )
    grouped["upper"] = (
        grouped["mean_com_distance_nm"]
        + grouped["sd_com_distance_nm"].fillna(0)
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
                variant_df["mean_com_distance_nm"],
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

        plt.xlabel("Time (ns)")
        plt.ylabel("COM Distance (nm)")
        plt.xlim(0, 300)
        plt.ylim(0.6,2.2)

        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"02_{comparison_group}_wt_vs_v8a_mean_sd_com_distance.png")


def plot_wt_vs_v8a_mean_com_distance_bars(replicate_summary: pd.DataFrame) -> None:
    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="mean_com_distance_nm",
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
            y="mean_com_distance_nm",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        plt.xlabel("Variant")
        plt.ylabel(f"Mean COM distance after {EQUILIBRATION_CUTOFF_NS} ns (nm)")

        save_plot(f"03_{comparison_group}_mean_com_distance_bar.png")


def plot_all_mean_com_distance_comparisons_one_figure(
    replicate_summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(18, 10))

    comparison_order = [
        "pro_mon_a",
        "mem_mon_a",
        "pro_dim_a",
        "pro_dim_b",
        "mem_dim_a",
        "mem_dim_b",
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
        y="mean_com_distance_nm",
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

    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(
        handles=handles,
        labels=["WT", "V8A"],
        loc="upper left",
        title=None
    )

    plt.xlabel("Comparison group")
    plt.ylabel(f"Mean COM distance (nm)")

    custom_labels = [
        "Soluble Monomer",
        "Membrane Monomer",
        "Soluble Dimer A",
        "Soluble Dimer B",
        "Membrane Dimer A",
        "Membrane Dimer B",
    ]

    plt.xticks(
        ticks=range(len(custom_labels)),
        labels=custom_labels,
        rotation=25,
        ha="right",
    )

    save_plot("04_all_groups_wt_vs_v8a_mean_com_distance.png")


def make_all_plots(df: pd.DataFrame, replicate_summary: pd.DataFrame) -> None:
    plot_wt_vs_v8a_replicate_pairs(df)
    plot_wt_vs_v8a_mean_sd(df)
    plot_wt_vs_v8a_mean_com_distance_bars(replicate_summary)
    plot_all_mean_com_distance_comparisons_one_figure(replicate_summary)


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