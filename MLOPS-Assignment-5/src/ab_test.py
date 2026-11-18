from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALPHA = 0.05
N_BOOTSTRAPS = 10_000

# Locate prediction CSV
candidates = (
    list(RESULTS.glob("*prediction*.csv"))
    + list(RESULTS.glob("*predictions*.csv"))
)
if not candidates:
    console.print("[bold red]No prediction CSV found in results/.[/bold red]")
    raise SystemExit(1)

# Prefer the file with the required columns
required = {
    "actual",
    "LogisticRegression_prediction",
    "RandomForest_prediction",
}
df = None

for path in dict.fromkeys(candidates):
    candidate_df = pd.read_csv(path)
    if required.issubset(candidate_df.columns):
        df = candidate_df
        break

if df is None:
    console.print(
        Panel(
            "Could not find a prediction CSV with these columns:\n"
            + "\n".join(sorted(required)),
            title="Prediction File Error",
            border_style="red",
        )
    )
    raise SystemExit(1)

# Read actual and predicted labels
y = df["actual"].to_numpy()
pred_lr = df["LogisticRegression_prediction"].to_numpy()
pred_rf = df["RandomForest_prediction"].to_numpy()

correct_lr = pred_lr == y
correct_rf = pred_rf == y

acc_lr = float(correct_lr.mean())
acc_rf = float(correct_rf.mean())
accuracy_diff = acc_rf - acc_lr

# McNemar's exact test
lr_correct_rf_wrong = int(np.sum(correct_lr & ~correct_rf))
lr_wrong_rf_correct = int(np.sum(~correct_lr & correct_rf))
discordant = lr_correct_rf_wrong + lr_wrong_rf_correct

p_value = (
    float(
        binomtest(
            min(lr_correct_rf_wrong, lr_wrong_rf_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    if discordant > 0
    else 1.0
)

# Paired bootstrap confidence interval
rng = np.random.default_rng(42)
n = len(y)
bootstrap_diffs = np.empty(N_BOOTSTRAPS)

for i in range(N_BOOTSTRAPS):
    idx = rng.integers(0, n, size=n)
    bootstrap_diffs[i] = (
        correct_rf[idx].mean() - correct_lr[idx].mean()
    )

ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])

# Header
console.print()
console.print(
    Panel(
        "[bold white]TITANIC CLASSIFICATION[/bold white]\n"
        "[cyan]A/B Testing & Statistical Evaluation[/cyan]",
        border_style="cyan",
        box=box.DOUBLE,
        expand=False,
    )
)

# Model performance
performance = Table(
    title="Model Performance",
    box=box.ROUNDED,
    header_style="bold cyan",
)
performance.add_column("Metric")
performance.add_column("Logistic Regression", justify="right")
performance.add_column("Random Forest", justify="right")

performance.add_row("Accuracy", f"{acc_lr:.4f} ({acc_lr:.2%})",
                    f"{acc_rf:.4f} ({acc_rf:.2%})")
performance.add_row(
    "Accuracy Difference",
    "—",
    f"{accuracy_diff:+.4f} ({accuracy_diff * 100:+.2f} pp)",
)
console.print(performance)

# Statistical results
statistics = Table(
    title="Statistical Test Results",
    box=box.ROUNDED,
    header_style="bold cyan",
)
statistics.add_column("Metric")
statistics.add_column("Result", justify="right")
statistics.add_row("McNemar Exact p-value", f"{p_value:.4f}")
statistics.add_row(
    "95% Bootstrap Confidence Interval",
    f"[{ci_lower:+.4f}, {ci_upper:+.4f}]",
)
statistics.add_row("Significance Level", f"{ALPHA:.2f}")
statistics.add_row("Bootstrap Resamples", f"{N_BOOTSTRAPS:,}")
console.print(statistics)

# Discordant predictions
discordance = Table(
    title="Discordant Predictions",
    box=box.ROUNDED,
    header_style="bold cyan",
)
discordance.add_column("Outcome")
discordance.add_column("Count", justify="right")
discordance.add_row("LR correct, RF wrong", str(lr_correct_rf_wrong))
discordance.add_row("LR wrong, RF correct", str(lr_wrong_rf_correct))
console.print(discordance)

# Conclusion
if p_value < ALPHA:
    conclusion = (
        "[bold green]Reject H0[/bold green]\n"
        "Evidence of a statistically significant difference "
        "in paired error rates."
    )
    style = "green"
else:
    conclusion = (
        "[bold yellow]Fail to reject H0[/bold yellow]\n"
        "Insufficient evidence of different error rates.\n"
        "The observed accuracy difference is not statistically "
        "significant at alpha = 0.05."
    )
    style = "yellow"

console.print(Panel(conclusion, title="Conclusion", border_style=style))
console.print(
    "[dim]Offline paired comparison on the same test set. "
    "Results may not generalize to new data.[/dim]"
)
console.print()