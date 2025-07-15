import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats

# Ensure full output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Tap Classification/TC Output'
WITHIN_RUN = 'within'
BETWEEN_RUN = 'between'

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def mean_and_ci(values, alpha=0.05):
    """Compute mean and 95% CI via Student's t-distribution."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    m = arr.mean()
    sem = stats.sem(arr, ddof=1)
    t = stats.t.ppf(1 - alpha/2, df=n-1)
    return m, m - t * sem, m + t * sem


def discover_lags(base_dir):
    """Identify all time-lag subdirectories for the first participant."""
    parts = [d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))]
    if not parts:
        return []
    sample = parts[0]
    lag_dirs = [d for d in os.listdir(os.path.join(base_dir, sample))
                if d.startswith('time_lag_')]
    return sorted(int(d.split('_')[-1]) for d in lag_dirs)

LAGS = discover_lags(BASE_DIR)


def load_within(pid, lag):
    """Return (BA, no_tap, tap) for within-subject at a given lag."""
    path = os.path.join(BASE_DIR, pid, f'time_lag_{lag}', WITHIN_RUN, 'results.pkl')
    if not os.path.isfile(path):
        return None
    data = pickle.load(open(path, 'rb'))['overall']
    return (data['mean_balanced_accuracy'],
            data['mean_true_notap_rate'],
            data['mean_true_tap_rate'])


def load_between(pid, lag):
    """Return lists of BA, no_tap, tap for between-subject at a given lag."""
    path = os.path.join(BASE_DIR, pid, f'time_lag_{lag}', BETWEEN_RUN, 'results.pkl')
    if not os.path.isfile(path):
        return None
    overall = pickle.load(open(path, 'rb'))['overall']
    bas, nts, tps = [], [], []
    for entry in overall.values():
        bas.append(entry['mean_balanced_accuracy'])
        nts.append(entry['mean_true_notap_rate'])
        tps.append(entry['mean_true_tap_rate'])
    return bas, nts, tps


def load_occlusion(pid, lag):
    """Return mean occlusion deltas list for a participant at a given lag."""
    path = os.path.join(BASE_DIR, pid, f'time_lag_{lag}', WITHIN_RUN, 'results.pkl')
    if not os.path.isfile(path):
        return None
    return pickle.load(open(path, 'rb'))['overall'].get('mean_occlusion_deltas')

# ── DATA COLLECTION ─────────────────────────────────────────────────────────────
participants = sorted([d for d in os.listdir(BASE_DIR)
                       if os.path.isdir(os.path.join(BASE_DIR, d))])

within_recs = []
between_recs = []
occl_recs = []

for pid in participants:
    # within-subject for each lag
    for lag in LAGS:
        res = load_within(pid, lag)
        if res:
            ba, nt, tt = res
            within_recs.append({
                'participant': pid,
                'lag': lag,
                'mean_balanced_accuracy': ba,
                'mean_true_notap_rate': nt,
                'mean_true_tap_rate': tt
            })
    # between-subject for each lag
    for lag in LAGS:
        b = load_between(pid, lag)
        if b:
            bas, nts, tps = b
            m_ba, lo_ba, hi_ba = mean_and_ci(bas)
            m_nt, lo_nt, hi_nt = mean_and_ci(nts)
            m_tt, lo_tt, hi_tt = mean_and_ci(tps)
            between_recs.append({
                'participant': pid,
                'lag': lag,
                'n_tests': len(bas),
                'mean_balanced_accuracy': m_ba,
                'CI95_lower': lo_ba,
                'CI95_upper': hi_ba,
                'mean_true_notap_rate': m_nt,
                'CI95_lower_notap': lo_nt,
                'CI95_upper_notap': hi_nt,
                'mean_true_tap_rate': m_tt,
                'CI95_lower_tap': lo_tt,
                'CI95_upper_tap': hi_tt
            })
    # occlusion across lags
    for lag in LAGS:
        oc = load_occlusion(pid, lag)
        if oc is not None:
            for cluster_idx, delta in enumerate(oc, start=1):
                occl_recs.append({
                    'lag': lag,
                    'cluster': f'Cluster {cluster_idx}',
                    'delta': delta
                })

# ── BUILD DATAFRAMES ────────────────────────────────────────────────────────────
within_df = pd.DataFrame(within_recs).set_index(['participant', 'lag'])
between_df = pd.DataFrame(between_recs).set_index(['participant', 'lag'])

# ── POOLED WITHIN-SUBJECT BY LAG ─────────────────────────────────────────────────
within_rows = []
for lag, grp in within_df.groupby(level='lag'):
    for metric in ['mean_balanced_accuracy',
                   'mean_true_notap_rate',
                   'mean_true_tap_rate']:
        arr = grp[metric].values
        m, lo, hi = mean_and_ci(arr)
        within_rows.append({
            'lag': lag,
            'metric': metric.replace('mean_', ''),
            'mean': m,
            'CI95_lower': lo,
            'CI95_upper': hi,
            'n_participants': len(arr)
        })
within_pooled = pd.DataFrame(within_rows)
within_pooled = within_pooled.set_index(['lag', 'metric']).unstack(level=-1)

# ── POOLED BETWEEN-SUBJECT BY LAG ───────────────────────────────────────────────
between_rows = []
for lag, grp in between_df.groupby(level='lag'):
    for metric in ['mean_balanced_accuracy',
                   'mean_true_notap_rate',
                   'mean_true_tap_rate']:
        arr = grp[metric].values
        m, lo, hi = mean_and_ci(arr)
        between_rows.append({
            'lag': lag,
            'metric': metric.replace('mean_', ''),
            'mean': m,
            'CI95_lower': lo,
            'CI95_upper': hi,
            'n_participants': len(arr)
        })
between_pooled = pd.DataFrame(between_rows)
between_pooled = between_pooled.set_index(['lag','metric']).unstack(level=-1)

# ── POOLED OCCLUSION ANALYSIS ──────────────────────────────────────────────────
occl_df = pd.DataFrame(occl_recs)
if not occl_df.empty:
    occl_summary = []
    for (lag, cluster), grp in occl_df.groupby(['lag', 'cluster']):
        m, lo, hi = mean_and_ci(grp['delta'].values)
        occl_summary.append({
            'lag': lag,
            'cluster': cluster,
            'mean': m,
            'CI95_lower': lo,
            'CI95_upper': hi,
            'n_participants': len(grp)
        })
    occl_pooled = pd.DataFrame(occl_summary)
    occl_pooled['cluster_num'] = occl_pooled['cluster'].str.extract(r'(\d+)').astype(int)
    occl_pooled = occl_pooled.sort_values(['lag', 'cluster_num'])
    occl_pooled = occl_pooled.drop(columns=['cluster_num']).set_index(['lag', 'cluster'])

# ── OUTPUT ──────────────────────────────────────────────────────────────────────
print("\nPooled within-participant results across participants:\n")
for metric in ['balanced_accuracy', 'true_notap_rate', 'true_tap_rate']:
    print(f"• {metric.replace('_', ' ').title()}:\n")
    sub = within_pooled.xs(metric, axis=1, level=1)
    print(sub)
    print("\n")

print("Pooled between-participant generalization across participants:\n")
for metric in ['balanced_accuracy', 'true_notap_rate', 'true_tap_rate']:
    print(f"• {metric.replace('_', ' ').title()}:\n")
    sub = between_pooled.xs(metric, axis=1, level=1)
    print(sub)
    print("\n")

print("Pooled occlusion analysis across participants:\n")
print(occl_pooled)
