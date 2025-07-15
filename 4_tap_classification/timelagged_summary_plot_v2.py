import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

MODE = 'within'   # 'within' or 'between'
SAVE_FIGURES = False
SHARE_YLIM = True

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR ='/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Tap Classification/TC Output'
OUTPUT_DIR = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Tap Classification'
METRICS = ['balanced_accuracy', 'true_notap_rate', 'true_tap_rate']
SEC_PER_LAG = 2.5

# ── PLOTTING FUNCTION ─────────────────────────────────────────────────────
def plot_accuracy_subplots(
    x,
    means_list,
    ci_lower_list,
    ci_upper_list,
    labels=('Balanced Accuracy', 'True NoTap Rate', 'True Tap Rate'),
    colors=('green', 'blue', 'red'),
    hline=0.5,
    hline_color='#FFA500',
    vline=0,
    vline_color='#D3D3D3',
    share_ylim=False,
    figsize=(6, 9)
):
    """
    Plot three vertically stacked accuracy subplots with error bars and reference lines.
    """
    # compute shared y-limits if requested
    if share_ylim:
        lowers = sum(ci_lower_list, [])
        uppers = sum(ci_upper_list, [])
        margin = (max(uppers) - min(lowers)) * 0.05
        ymin = min(lowers) - margin
        ymax = max(uppers) + margin
    else:
        ymin = ymax = None

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=figsize)

    for ax, means, low, high, color, label in zip(
        axs, means_list, ci_lower_list, ci_upper_list, colors, labels
    ):
        ax.axvline(vline, color=vline_color, linestyle='--', zorder=0)
        ax.errorbar(
            x, means,
            yerr=[
                [m - l for m, l in zip(means, low)],
                [h - m for h, m in zip(high, means)]
            ],
            fmt='o-', capsize=5, capthick=1.5,
            color=color, zorder=2
        )
        # autoscale to the data-only range
        ax.relim()
        ax.autoscale_view()
        # if sharing y-limits, set them before drawing the chance line
        if share_ylim and ymin is not None:
            ax.set_ylim(ymin, ymax)
            lower, upper = ymin, ymax
        else:
            # otherwise use the individual subplot limits
            lower, upper = ax.get_ylim()
        # draw chance line only if within the chosen limits
        if lower <= hline <= upper:
            ax.axhline(hline, color=hline_color, linestyle='--', zorder=1)
        ax.set_ylabel('Accuracy')
        ax.set_title(label)

    axs[-1].set_xlabel('Time Lag [s]')
    plt.tight_layout()
    return fig, axs

# ── HELPER ─────────────────────────────────────────────────────────────────────
def mean_and_ci(arr, alpha=0.05):
    a   = np.asarray(arr, dtype=float)
    m   = a.mean()
    sem = stats.sem(a, ddof=1)
    t   = stats.t.ppf(1 - alpha/2, df=a.size - 1)
    return m, m - t*sem, m + t*sem

# ── DISCOVER PARTICIPANTS & LAGS ───────────────────────────────────────────────
participants = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
)
if participants:
    sample   = participants[0]
    lag_dirs = [d for d in os.listdir(os.path.join(BASE_DIR, sample))
                if d.startswith('time_lag_')]
    LAGS     = sorted(int(d.split('_')[-1]) for d in lag_dirs)
else:
    LAGS = []

# ── COLLECT PER-PARTICIPANT RESULTS ────────────────────────────────────────────
records = []
for pid in participants:
    for lag in LAGS:
        fn = os.path.join(BASE_DIR, pid, f'time_lag_{lag}', MODE, 'results.pkl')
        if not os.path.isfile(fn):
            continue
        with open(fn, 'rb') as f:
            res = pickle.load(f)
        overall = res.get('overall', {})
        # within-subject: overall is a dict of metrics
        if MODE == 'within':
            if not all(k in overall for k in (
                'mean_balanced_accuracy',
                'mean_true_notap_rate',
                'mean_true_tap_rate'
            )):
                continue
            ba = overall['mean_balanced_accuracy']
            nt = overall['mean_true_notap_rate']
            tt = overall['mean_true_tap_rate']
        # between-subject: overall maps test_pid → metrics
        else:
            vals = list(overall.values())
            if not vals:
                continue
            ba = np.mean([v['mean_balanced_accuracy'] for v in vals])
            nt = np.mean([v['mean_true_notap_rate']   for v in vals])
            tt = np.mean([v['mean_true_tap_rate']     for v in vals])
        records.append({
            'lag':               lag,
            'balanced_accuracy': ba,
            'true_notap_rate':   nt,
            'true_tap_rate':     tt
        })

df = pd.DataFrame(records)

# ── POOL ACROSS PARTICIPANTS BY LAG ────────────────────────────────────────────
lags_sorted = sorted(df['lag'].unique())
x_vals     = [lag * SEC_PER_LAG for lag in lags_sorted]

ba_means = []; ba_lo = []; ba_hi = []
nt_means = []; nt_lo = []; nt_hi = []
t_means  = []; t_lo  = []; t_hi  = []

for lag in lags_sorted:
    grp = df[df['lag'] == lag]
    for col, means, lo, hi in [
        ('balanced_accuracy', ba_means, ba_lo, ba_hi),
        ('true_notap_rate',   nt_means, nt_lo, nt_hi),
        ('true_tap_rate',     t_means,  t_lo,  t_hi)
    ]:
        arr = grp[col].values
        m, l, u = mean_and_ci(arr)
        means.append(m); lo.append(l); hi.append(u)

# ── FINAL SUMMARY PLOT ────────────────────────────────────────────────────────
fig, axs = plot_accuracy_subplots(
    x              = x_vals,
    means_list     = [ba_means, nt_means, t_means],
    ci_lower_list  = [ba_lo,    nt_lo,    t_lo   ],
    ci_upper_list  = [ba_hi,    nt_hi,    t_hi   ],
    share_ylim     = SHARE_YLIM,
)

plt.show()

if SAVE_FIGURES:
    fig.savefig(os.path.join(OUTPUT_DIR, 'timelagged_accuracy.svg'), bbox_inches='tight')
