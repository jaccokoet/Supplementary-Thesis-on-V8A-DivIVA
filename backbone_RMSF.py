from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


# -----------------------------
# Settings
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_files"
GRAPH_DIR = BASE_DIR / "graphs"
OUTPUT_DIR = BASE_DIR / "output_files"

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
# Colors
# -----------------------------
light_pink = "#ff98cb"
dark_pink = "#ec3499"
light_blue = "#65a9ed"
dark_blue = "#1546c7"

CHAIN_VARIANT_COLORS = {
    ("wt", "chain_a"): dark_blue,
    ("wt", "chain_b"): light_blue,
    ("v8a", "chain_a"): dark_pink,
    ("v8a", "chain_b"): light_pink,
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
        pro_mon_wt_repl1_protein_backbone_rmsf.xvg
        pro_mon_v8a_repl2_protein_backbone_rmsf.xvg

        pro_dim_wt_repl1_chain_a_backbone_rmsf.xvg
        pro_dim_wt_repl1_chain_b_backbone_rmsf.xvg
        pro_dim_v8a_repl1_chain_a_backbone_rmsf.xvg
        pro_dim_v8a_repl1_chain_b_backbone_rmsf.xvg

        mem_dim_wt_repl1_chain_a_backbone_rmsf.xvg
        mem_dim_v8a_repl1_chain_b_backbone_rmsf.xvg
    """

    stem = filename.replace("_backbone_rmsf.xvg", "")

    pattern = re.compile(
        r"^(?P<environment>mem|pro)_"
        r"(?P<oligomer>mon|dim)_"
        r"(?P<variant>wt|v8a)_"
        r"(?P<replicate>repl\d+)_"
        r"(?P<selection>protein|chain_a|chain_b)$"
    )

    match = pattern.match(stem)

    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    data = match.groupdict()
    data["system"] = f"{data['environment']}_{data['oligomer']}"
    data["condition"] = (
        f"{data['system']}_{data['variant']}_"
        f"{data['replicate']}_{data['selection']}"
    )

    return data


def read_xvg(filepath: Path) -> pd.DataFrame:
    """
    Read a GROMACS gmx rmsf .xvg file.
    Uses the first two numeric columns as residue and RMSF.
    """

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
                residue = int(float(parts[0]))
                rmsf = float(parts[1])
                rows.append((residue, rmsf))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric RMSF data found in {filepath}")

    return pd.DataFrame(rows, columns=["residue", "rmsf_nm"])


def load_all_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_backbone_rmsf.xvg"))

    if not files:
        raise FileNotFoundError(f"No *_backbone_rmsf.xvg files found in {INPUT_DIR}")

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

    combined["chain"] = combined["selection"].map({
        "protein": "Protein",
        "chain_a": "Chain A",
        "chain_b": "Chain B",
    })

    combined = combined.sort_values(
        ["system", "variant", "replicate_number", "selection", "residue"]
    ).reset_index(drop=True)

    return combined


# -----------------------------
# Summary tables
# -----------------------------
def calculate_summary_by_file(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "system",
                "environment",
                "oligomer",
                "variant",
                "replicate",
                "selection",
                "chain",
                "condition",
                "source_file",
            ],
            as_index=False,
        )
        .agg(
            n_residues=("rmsf_nm", "size"),
            start_residue=("residue", "min"),
            end_residue=("residue", "max"),
            mean_rmsf_nm=("rmsf_nm", "mean"),
            median_rmsf_nm=("rmsf_nm", "median"),
            sd_rmsf_nm=("rmsf_nm", "std"),
            min_rmsf_nm=("rmsf_nm", "min"),
            max_rmsf_nm=("rmsf_nm", "max"),
        )
        .sort_values(["system", "variant", "replicate", "selection"])
    )


def calculate_residue_summary(df: pd.DataFrame) -> pd.DataFrame:
    residue_summary = (
        df.groupby(
            ["system", "oligomer", "variant", "selection", "chain", "residue"],
            as_index=False,
        )
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_rmsf_nm=("rmsf_nm", "mean"),
            sd_rmsf_nm=("rmsf_nm", "std"),
            min_rmsf_nm=("rmsf_nm", "min"),
            max_rmsf_nm=("rmsf_nm", "max"),
        )
        .sort_values(["system", "variant", "selection", "residue"])
    )

    residue_summary["lower"] = (
        residue_summary["mean_rmsf_nm"] -
        residue_summary["sd_rmsf_nm"].fillna(0)
    )

    residue_summary["upper"] = (
        residue_summary["mean_rmsf_nm"] +
        residue_summary["sd_rmsf_nm"].fillna(0)
    )

    return residue_summary


# -----------------------------
# Per-residue t-tests
# -----------------------------
def calculate_per_residue_t_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-residue Welch t-tests.

    Monomers:
        WT protein vs V8A protein

    Dimers:
        WT chain A vs V8A chain B
        WT chain B vs V8A chain B
    """

    rows = []

    tests = [
        {
            "test_name": "monomer_wt_protein_vs_v8a_protein",
            "oligomer": "mon",
            "wt_selection": "protein",
            "v8a_selection": "protein",
        },
        {
            "test_name": "dimer_wt_chain_a_vs_v8a_chain_a",
            "oligomer": "dim",
            "wt_selection": "chain_a",
            "v8a_selection": "chain_a",
        },
        {
            "test_name": "dimer_wt_chain_b_vs_v8a_chain_b",
            "oligomer": "dim",
            "wt_selection": "chain_b",
            "v8a_selection": "chain_b",
        },
    ]

    for test in tests:
        test_df = df[df["oligomer"] == test["oligomer"]].copy()

        for system in sorted(test_df["system"].unique()):
            system_df = test_df[test_df["system"] == system]

            residues = sorted(system_df["residue"].unique())

            for residue in residues:
                wt_values = system_df[
                    (system_df["variant"] == "wt") &
                    (system_df["selection"] == test["wt_selection"]) &
                    (system_df["residue"] == residue)
                ]["rmsf_nm"]

                v8a_values = system_df[
                    (system_df["variant"] == "v8a") &
                    (system_df["selection"] == test["v8a_selection"]) &
                    (system_df["residue"] == residue)
                ]["rmsf_nm"]

                if len(wt_values) < 2 or len(v8a_values) < 2:
                    continue

                t_statistic, p_value = stats.ttest_ind(
                    wt_values,
                    v8a_values,
                    equal_var=False,
                )

                wt_mean = wt_values.mean()
                v8a_mean = v8a_values.mean()
                wt_sd = wt_values.std(ddof=1)
                v8a_sd = v8a_values.std(ddof=1)

                rows.append({
                    "test_name": test["test_name"],
                    "system": system,
                    "residue": residue,

                    "chain": test["v8a_selection"],

                    "wt_mean_rmsf_nm": wt_mean,
                    "wt_sd_rmsf_nm": wt_sd,

                    "v8a_mean_rmsf_nm": v8a_mean,
                    "v8a_sd_rmsf_nm": v8a_sd,

                    "difference_v8a_minus_wt_nm": v8a_mean - wt_mean,
                    "absolute_difference_nm": abs(v8a_mean - wt_mean),

                    "t_statistic": t_statistic,
                    "p_value": p_value,
                    "significant_p_lt_0_05": p_value < 0.05,
                })

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    results["fdr_adjusted_p_value"] = pd.NA
    results["significant_fdr_0_05"] = False

    for test_name, sub in results.groupby("test_name"):
        corrected = multipletests(
            sub["p_value"],
            alpha=0.05,
            method="fdr_bh",
        )

        results.loc[sub.index, "fdr_adjusted_p_value"] = corrected[1]
        results.loc[sub.index, "significant_fdr_0_05"] = corrected[0]

    results = results.sort_values(
        ["test_name", "system", "residue"]
    ).reset_index(drop=True)

    return results


# -----------------------------
# Save tables
# -----------------------------
def save_tables(
    df: pd.DataFrame,
    summary_by_file: pd.DataFrame,
    residue_summary: pd.DataFrame,
    per_residue_t_tests: pd.DataFrame,
) -> None:
    df.to_csv(
        OUTPUT_DIR / "all_backbone_rmsf_long_format.csv",
        index=False,
    )

    summary_by_file.to_csv(
        OUTPUT_DIR / "backbone_rmsf_summary_by_file.csv",
        index=False,
    )

    residue_summary.to_csv(
        OUTPUT_DIR / "backbone_rmsf_summary_by_residue.csv",
        index=False,
    )

    per_residue_t_tests.to_csv(
        OUTPUT_DIR / "per_residue_backbone_rmsf_t_tests.csv",
        index=False,
    )


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_monomers_per_replicate(df: pd.DataFrame) -> None:

    MONOMER_COLORS = {
        "wt": dark_blue,
        "v8a": dark_pink,
    }

    plot_data = df[
        (df["oligomer"] == "mon") &
        (df["selection"] == "protein")
    ].copy()

    for (system, replicate), sub in plot_data.groupby(["system", "replicate"]):

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:

            line_df = sub[
                sub["variant"] == variant
            ].sort_values("residue")

            if line_df.empty:
                continue

            plt.plot(
                line_df["residue"],
                line_df["rmsf_nm"],
                color=MONOMER_COLORS[variant],
                label=f"{variant.upper()}",
                linewidth=1,
            )

        plt.xlabel("Residue")
        plt.ylabel("Backbone RMSF (nm)")

        plt.xlim(0, 60)
        plt.ylim(0, 1.6)

        #legend = plt.legend(
        #    loc="upper left",
        #)
        legend = plt.legend(title=f"{replicate.upper()}", loc="upper left")


        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"01_{system}_{replicate}_monomer_backbone_rmsf.png"
        )

def plot_monomers_mean_sd(residue_summary: pd.DataFrame) -> None:

    MONOMER_COLORS = {
        "wt": dark_blue,
        "v8a": dark_pink,
    }

    plot_data = residue_summary[
        (residue_summary["oligomer"] == "mon") &
        (residue_summary["selection"] == "protein")
    ].copy()

    for system, sub in plot_data.groupby("system"):

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:

            line_df = sub[
                sub["variant"] == variant
            ].sort_values("residue")

            if line_df.empty:
                continue

            plt.plot(
                line_df["residue"],
                line_df["mean_rmsf_nm"],
                color=MONOMER_COLORS[variant],
                label=f"{variant.upper()}",
                linewidth=1,
            )

            plt.fill_between(
                line_df["residue"],
                line_df["lower"],
                line_df["upper"],
                color=MONOMER_COLORS[variant],
                alpha=0.15,
            )

        plt.xlabel("Residue")
        plt.ylabel("Backbone RMSF (nm)")

        plt.xlim(0, 60)
        plt.ylim(0, 1.6)

        legend = plt.legend(
            loc="upper left",
        )

        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"02_{system}_monomer_mean_sd_backbone_rmsf.png"
        )

def plot_dimer_chains_per_replicate(df: pd.DataFrame) -> None:
    """
    One graph per dimer system and replicate.
    Shows:
        WT chain A  = dark blue
        WT chain B  = light blue
        V8A chain A = dark pink
        V8A chain B = light pink
    """

    plot_data = df[
        (df["oligomer"] == "dim") &
        (df["selection"].isin(["chain_a", "chain_b"]))
    ].copy()

    for (system, replicate), sub in plot_data.groupby(["system", "replicate"]):
        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            for selection in ["chain_a", "chain_b"]:
                line_df = sub[
                    (sub["variant"] == variant) &
                    (sub["selection"] == selection)
                ].sort_values("residue")

                if line_df.empty:
                    continue

                label = f"{variant.upper()} {line_df['chain'].iloc[0]}"

                plt.plot(
                    line_df["residue"],
                    line_df["rmsf_nm"],
                    label=label,
                    color=CHAIN_VARIANT_COLORS[(variant, selection)],
                    linewidth=1,
                )

        plt.xlabel("Residue")
        plt.ylabel("Backbone RMSF (nm)")
        plt.xlim(0, 60)
        plt.ylim(0,1.6)

        #legend = plt.legend(
        #    #title=f"{system.upper()} {replicate.upper()}",
        #    loc="upper left",
        #)
        legend = plt.legend(title=f"{replicate.upper()}", loc="upper left")


        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"03_{system}_{replicate}_dimer_chains_backbone_rmsf.png"
        )


def plot_dimer_chains_mean_sd(residue_summary: pd.DataFrame) -> None:
    """
    One graph per dimer system.
    Shows replicate mean ± SD for:
        WT chain A  = dark blue
        WT chain B  = light blue
        V8A chain A = dark pink
        V8A chain B = light pink
    """

    plot_data = residue_summary[
        (residue_summary["oligomer"] == "dim") &
        (residue_summary["selection"].isin(["chain_a", "chain_b"]))
    ].copy()

    for system, sub in plot_data.groupby("system"):
        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            for selection in ["chain_a", "chain_b"]:
                line_df = sub[
                    (sub["variant"] == variant) &
                    (sub["selection"] == selection)
                ].sort_values("residue")

                if line_df.empty:
                    continue

                label = f"{variant.upper()} {line_df['chain'].iloc[0]}"

                plt.plot(
                    line_df["residue"],
                    line_df["mean_rmsf_nm"],
                    label=label,
                    color=CHAIN_VARIANT_COLORS[(variant, selection)],
                    linewidth=1,
                )

                plt.fill_between(
                    line_df["residue"],
                    line_df["lower"],
                    line_df["upper"],
                    color=CHAIN_VARIANT_COLORS[(variant, selection)],
                    alpha=0.15,
                )

        plt.xlabel("Residue")
        plt.ylabel("Backbone RMSF (nm)")
        plt.xlim(0, 60)
        plt.ylim(0,1.6)

        legend = plt.legend(
            #title=system.upper(),
            loc="upper left",
        )


        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"04_{system}_dimer_chains_mean_sd_backbone_rmsf.png"
        )


def make_all_plots(
    df: pd.DataFrame,
    residue_summary: pd.DataFrame,
) -> None:
    plot_dimer_chains_per_replicate(df)
    plot_dimer_chains_mean_sd(residue_summary)
    plot_monomers_per_replicate(df)
    plot_monomers_mean_sd(residue_summary)


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    make_dirs()

    print(f"Reading XVG files from: {INPUT_DIR}")

    df = load_all_data()

    print(f"Loaded {df['source_file'].nunique()} files")
    print(f"Total data points: {len(df):,}")

    summary_by_file = calculate_summary_by_file(df)
    residue_summary = calculate_residue_summary(df)
    per_residue_t_tests = calculate_per_residue_t_tests(df)

    save_tables(
        df,
        summary_by_file,
        residue_summary,
        per_residue_t_tests,
    )

    make_all_plots(df, residue_summary)

    print("\nDone!")


if __name__ == "__main__":
    main()