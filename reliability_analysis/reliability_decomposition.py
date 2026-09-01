"""
FISAD Reliability Decomposition
==================================
Estimates the inter-rater reliability of the crowd-sourced Big Five trait
scores, and projects the annotation density needed to reach higher
reliability levels via the Spearman-Brown prophecy formula.

Method: each image's trait score is an aggregate proportion of pairwise
"wins" out of N_i comparisons. Treating this as a binomial proportion, the
expected sampling variance for image i is p_i(1-p_i)/N_i. Reliability is
estimated as the proportion of total between-image variance NOT
attributable to this sampling noise (analogous to an intraclass
correlation coefficient):

    Reliability = (Total Variance - Mean Sampling Variance) / Total Variance

Usage:
    python3 reliability_decomposition.py --input FISAD_data.csv
"""
import argparse
import numpy as np
import pandas as pd

TRAITS = {'a': 'Agreeableness', 'e': 'Extraversion', 'o': 'Openness',
          'c': 'Conscientiousness', 'n': 'Neuroticism'}


def estimate_reliability(p, n):
    """Estimate reliability (ICC-analogous) for a single trait's scores.

    Args:
        p: array of per-image trait scores (proportions, 0-1)
        n: array of per-image comparison counts (No_compare)

    Returns:
        (reliability, total_variance, mean_sampling_variance)
    """
    total_var = np.var(p, ddof=1)
    sampling_var = p * (1 - p) / n
    mean_sampling_var = np.mean(sampling_var)
    reliability = (total_var - mean_sampling_var) / total_var
    return reliability, total_var, mean_sampling_var


def fit_beta_moments(p, n):
    """Method-of-moments Beta(alpha, beta) prior for empirical Bayes shrinkage,
    correcting observed variance for known binomial sampling noise."""
    mean_p = np.mean(p)
    var_p = np.var(p, ddof=1)
    mean_sampling_var = np.mean(p * (1 - p) / n)
    true_var = max(var_p - mean_sampling_var, 1e-6)
    common = mean_p * (1 - mean_p) / true_var - 1
    alpha = max(mean_p * common, 0.1)
    beta = max((1 - mean_p) * common, 0.1)
    return alpha, beta


def empirical_bayes_shrinkage_reliability(p, n):
    """Reliability of Beta-Binomial posterior-mean shrunk estimates, as a
    check on whether improved point-estimation (rather than more data)
    can recover reliability."""
    alpha, beta = fit_beta_moments(p, n)
    successes = p * n
    shrunk = (successes + alpha) / (n + alpha + beta)
    shrunk_var_total = np.var(shrunk, ddof=1)
    shrinkage_factor = n / (n + alpha + beta)
    shrunk_sampling_var = np.mean((shrinkage_factor ** 2) * (p * (1 - p) / n))
    return max(0, (shrunk_var_total - shrunk_sampling_var) / shrunk_var_total)


def spearman_brown_projection(r_current, n_current, target_reliabilities):
    """Projects the number of raters/comparisons needed to reach each
    target reliability level, given the current reliability at n_current."""
    projections = {}
    for r_target in target_reliabilities:
        k = (r_target * (1 - r_current)) / (r_current * (1 - r_target))
        projections[r_target] = n_current * k
    return projections


def main():
    parser = argparse.ArgumentParser(description="FISAD reliability decomposition")
    parser.add_argument("--input", default="FISAD_data.csv", help="Path to FISAD_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n = df["No_compare"].values

    print(f"{'Trait':20s} {'Reliability':>12s} {'EB-shrunk Rel.':>16s}")
    reliabilities = {}
    for code, name in TRAITS.items():
        p = df[code].values
        rel, total_var, sampling_var = estimate_reliability(p, n)
        eb_rel = empirical_bayes_shrinkage_reliability(p, n)
        reliabilities[code] = rel
        print(f"{name:20s} {rel:12.3f} {eb_rel:16.3f}")

    mean_n = np.mean(n)
    mean_rel = np.mean(list(reliabilities.values()))
    print(f"\nMean reliability across traits: {mean_rel:.3f} (at mean {mean_n:.2f} raters/image)")

    print("\n=== Spearman-Brown projection (using mean current reliability) ===")
    projections = spearman_brown_projection(mean_rel, mean_n, [0.3, 0.5, 0.7, 0.8, 0.9])
    for target, raters_needed in projections.items():
        print(f"  Target reliability {target}: ~{raters_needed:.0f} raters/image needed")


if __name__ == "__main__":
    main()
