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

# Header says Time (ps)
CONVERT_PS_TO_NS = True

EQUILIBRATION_CUTOFF_NS = 50

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
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
        mem_mon_wt_repl1_tilt_angle_z.xvg
        mem_mon_v8a_repl3_tilt_angle_z.xvg
        mem_dim_wt_repl1_tilt_angle_z.xvg
        mem_dim_v8a_repl3_tilt_angle_z.xvg
    """
    stem = filename.replace("_tilt_angle_z.xvg", "")

    pattern = re.compile(
        r"^(?P<environment>mem)_"
        r"(?P<oligomer>mon|dim)_"
        r"(?P<variant>wt|v8a)_"
        r"(?P<replicate>repl\d+)$"
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
                tilt_angle = float(parts[1])
                rows.append((time, tilt_angle))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric tilt-angle data found in {filepath}")

    df = pd.DataFrame(rows, columns=["time_raw", "tilt_angle_deg"])
    df["time_ns"] = df["time_raw"] / 1000 if CONVERT_PS_TO_NS else df["time_raw"]

    return df


def load_all_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_tilt_angle_z.xvg"))

    if not files:
        raise FileNotFoundError(f"No *_tilt_angle_z.xvg files found in {INPUT_DIR}")

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
            n_points=("tilt_angle_deg", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_tilt_angle_deg=("tilt_angle_deg", "mean"),
            median_tilt_angle_deg=("tilt_angle_deg", "median"),
            std_tilt_angle_deg=("tilt_angle_deg", "std"),
            min_tilt_angle_deg=("tilt_angle_deg", "min"),
            max_tilt_angle_deg=("tilt_angle_deg", "max"),
        )
        .sort_values(["system", "variant", "replicate"])
    )

    return summary


def calculate_wt_vs_v8a_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        replicate_summary.groupby(
            ["system", "comparison_group", "variant"],
            as_index=False,
        )
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_tilt_angle_deg=("mean_tilt_angle_deg", "mean"),
            sd_mean_tilt_angle_deg=("mean_tilt_angle_deg", "std"),
            mean_max_tilt_angle_deg=("max_tilt_angle_deg", "mean"),
            sd_max_tilt_angle_deg=("max_tilt_angle_deg", "std"),
        )
        .sort_values(["system", "variant"])
    )


def calculate_wt_vs_v8a_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = replicate_summary.pivot_table(
        index=["system", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_tilt_angle_deg", "max_tilt_angle_deg"],
    )

    rows = []

    for index, row in pivot.iterrows():
        system, comparison_group, replicate = index

        try:
            rows.append({
                "system": system,
                "comparison_group": comparison_group,
                "replicate": replicate,
                "mean_tilt_angle_difference_v8a_minus_wt_deg":
                    row[("mean_tilt_angle_deg", "v8a")]
                    - row[("mean_tilt_angle_deg", "wt")],
                "max_tilt_angle_difference_v8a_minus_wt_deg":
                    row[("max_tilt_angle_deg", "v8a")]
                    - row[("max_tilt_angle_deg", "wt")],
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
            wt["mean_tilt_angle_deg"],
            v8a["mean_tilt_angle_deg"],
            equal_var=False,
        )

        rows.append({
            "comparison_group": comparison_group,
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),
            "wt_mean_tilt_angle_deg": wt["mean_tilt_angle_deg"].mean(),
            "wt_sd_tilt_angle_deg": wt["mean_tilt_angle_deg"].std(),
            "v8a_mean_tilt_angle_deg": v8a["mean_tilt_angle_deg"].mean(),
            "v8a_sd_tilt_angle_deg": v8a["mean_tilt_angle_deg"].std(),
            "mean_tilt_angle_t_statistic": mean_t,
            "mean_tilt_angle_p_value": mean_p,
            "mean_tilt_angle_significant_p_lt_0_05": mean_p < 0.05,
        })

    return pd.DataFrame(rows)


def save_tables(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
) -> None:
    df.to_csv(OUTPUT_DIR / "all_tilt_angle_long_format.csv", index=False)
    replicate_summary.to_csv(OUTPUT_DIR / "tilt_angle_summary_by_replicate.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "wt_vs_v8a_tilt_angle_summary_by_group.csv", index=False)
    differences.to_csv(OUTPUT_DIR / "wt_vs_v8a_tilt_angle_differences_by_replicate.csv", index=False)
    t_tests.to_csv(OUTPUT_DIR / "wt_vs_v8a_tilt_angle_t_tests.csv", index=False)


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_wt_vs_v8a_replicate_pairs(df: pd.DataFrame) -> None:
    for (comparison_group, replicate), sub in df.groupby(
        ["comparison_group", "replicate"]
    ):
        if not {"wt", "v8a"}.issubset(set(sub["variant"])):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["tilt_angle_deg"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel("Tilt angle to membrane normal (degrees)")
        plt.xlim(0, 300)
        plt.ylim(0, 180)

        legend = plt.legend(title=replicate.upper(), loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"01_{comparison_group}_{replicate}_wt_vs_v8a_tilt_angle.png")


def plot_wt_vs_v8a_mean_sd(df: pd.DataFrame) -> None:

    plot_df = df[
        ~(
            (df["comparison_group"] == "mem_dim")
            & (df["variant"] == "wt")
            & (df["replicate"] == "repl1")
        )
    ].copy()

    grouped = (
        plot_df.groupby(
            ["comparison_group", "variant", "time_ns"],
            as_index=False,
        )
        .agg(
            mean_tilt_angle_deg=("tilt_angle_deg", "mean"),
            sd_tilt_angle_deg=("tilt_angle_deg", "std"),
        )
    )

    grouped["lower"] = (
        grouped["mean_tilt_angle_deg"]
        - grouped["sd_tilt_angle_deg"].fillna(0)
    )

    grouped["upper"] = (
        grouped["mean_tilt_angle_deg"]
        + grouped["sd_tilt_angle_deg"].fillna(0)
    )

    for comparison_group, sub in grouped.groupby("comparison_group"):
        if not {"wt", "v8a"}.issubset(set(sub["variant"])):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = (
                sub[sub["variant"] == variant]
                .sort_values("time_ns")
            )

            plt.plot(
                variant_df["time_ns"],
                variant_df["mean_tilt_angle_deg"],
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
        plt.ylabel("Tilt angle to membrane normal (degrees)")
        plt.xlim(0, 300)
        plt.ylim(0, 180)

        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"02_{comparison_group}_wt_vs_v8a_mean_sd_tilt_angle.png"
        )

def plot_wt_vs_v8a_mean_tilt_angle_bars(replicate_summary: pd.DataFrame) -> None:
    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        if not {"wt", "v8a"}.issubset(set(sub["variant"])):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="mean_tilt_angle_deg",
            errorbar="sd",
            capsize=0.15,
            order=["wt", "v8a"],
            palette=[dark_blue, dark_pink],
            err_kws={"color": "black", "linewidth": 2},
        )

        sns.stripplot(
            data=sub,
            x="variant",
            y="mean_tilt_angle_deg",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        plt.xlabel("Variant")
        plt.ylabel(f"Mean tilt angle after {EQUILIBRATION_CUTOFF_NS} ns (degrees)")
        plt.ylim(0, 180)

        save_plot(f"03_{comparison_group}_mean_tilt_angle_bar.png")


def plot_all_mean_tilt_angle_comparisons_one_figure(
    replicate_summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(12, 10))

    comparison_order = ["mem_mon", "mem_dim"]

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
        y="mean_tilt_angle_deg",
        hue="variant",
        errorbar="sd",
        capsize=0.15,
        hue_order=["wt", "v8a"],
        palette={"wt": dark_blue, "v8a": dark_pink},
        err_kws={"color": "black", "linewidth": 2},
    )

    plt.legend(labels=["WT", "V8A"], loc="upper left", title=None)

    plt.xlabel("Comparison group")
    plt.ylabel(f"Mean tilt angle after {EQUILIBRATION_CUTOFF_NS} ns (degrees)")
    plt.ylim(0, 180)

    plt.xticks(
        ticks=range(2),
        labels=["Membrane Monomer", "Membrane Dimer"],
        rotation=25,
        ha="right",
    )

    save_plot("04_all_groups_wt_vs_v8a_mean_tilt_angle.png")


def make_all_plots(df: pd.DataFrame, replicate_summary: pd.DataFrame) -> None:
    plot_wt_vs_v8a_replicate_pairs(df)
    plot_wt_vs_v8a_mean_sd(df)
    plot_wt_vs_v8a_mean_tilt_angle_bars(replicate_summary)
    plot_all_mean_tilt_angle_comparisons_one_figure(replicate_summary)


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