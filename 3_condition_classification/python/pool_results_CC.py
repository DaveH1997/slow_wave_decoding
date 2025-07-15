import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats

# ── CONFIG ──────────────────────────────────────────────────────────────────────
# BASE_DIR           = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Condition Classification/Python/CC Output/Raw'
BASE_DIR           = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Condition Classification/Python/CC Output/Cleaned'
WITHIN_RUN         = 'within'
BETWEEN_RUN        = 'between'
WITHIN_TEMPORD_RUN = 'within_temp_ord'

# ── HELPERS ────────────────────────────────────────────────────────────────────
def mean_and_ci(arr, alpha=0.05):
    arr = np.array(arr)
    n   = arr.size
    mean = arr.mean()
    sem  = stats.sem(arr, ddof=1)
    t    = stats.t.ppf(1 - alpha/2, df=n-1)
    return mean, mean - t*sem, mean + t*sem

# ── LOAD FUNCTIONS ─────────────────────────────────────────────────────────────
def load_within_lists(pid):
    p = os.path.join(BASE_DIR, pid, WITHIN_RUN, 'results.pkl')
    if not os.path.exists(p): return None
    folds = pickle.load(open(p,'rb'))['folds']
    ba = [f['balanced_accuracy'] for f in folds]
    tm = [f['true_movie_rate']   for f in folds]
    tp = [f['true_phone_rate']   for f in folds]
    return ba, tm, tp

def load_between_lists(pid):
    p = os.path.join(BASE_DIR, pid, BETWEEN_RUN, 'results.pkl')
    if not os.path.exists(p): return None
    overall = pickle.load(open(p,'rb'))['overall']
    ba, tm, tp = [], [], []
    for s in overall.values():
        ba.append(s['mean_balanced_accuracy'])
        tm.append(s['mean_true_movie_rate'])
        tp.append(s['mean_true_phone_rate'])
    return ba, tm, tp

def load_within_tempord(pid):
    p = os.path.join(BASE_DIR, pid, WITHIN_TEMPORD_RUN, 'results.pkl')
    if not os.path.exists(p): return None
    folds = pickle.load(open(p,'rb'))['folds']
    ba = [f['balanced_accuracy'] for f in folds]
    bd = [f['block_distance']     for f in folds]
    return ba, bd

def load_between_tempord(pid):
    p = os.path.join(BASE_DIR, pid, BETWEEN_RUN, 'results.pkl')
    if not os.path.exists(p): return None
    overall = pickle.load(open(p,'rb'))['overall']
    # get distances from any entry
    sample = next(iter(overall.values()))
    bd = sample['block_distance']
    # stack each test‐participant's temp‐ord BAs
    mat = np.vstack([s['mean_balanced_accuracy_temp_ord'] for s in overall.values()])
    return mat.mean(axis=0), bd

def load_within_occlusion(pid):
    p = os.path.join(BASE_DIR, pid, WITHIN_RUN, 'results.pkl')
    if not os.path.exists(p): return None
    overall = pickle.load(open(p,'rb'))['overall']
    return overall.get('mean_occlusion_deltas', None)

# ── COLLECT METRICS ─────────────────────────────────────────────────────────────
within_rows            = []
between_rows           = []
within_tempord_data    = {}
between_tempord_data   = {}
within_occl_data       = {}

for pid in sorted(os.listdir(BASE_DIR)):
    # within‐participant
    w = load_within_lists(pid)
    if w:
        ba, tm, tp = w
        within_rows.append({
            'participant': pid,
            'n_folds': len(ba),
            **dict(zip(
                ['mean_balanced_accuracy','CI95_lower_ba','CI95_upper_ba'],
                mean_and_ci(ba)
            )),
            **dict(zip(
                ['mean_true_movie_rate','CI95_lower_movie','CI95_upper_movie'],
                mean_and_ci(tm)
            )),
            **dict(zip(
                ['mean_true_phone_rate','CI95_lower_phone','CI95_upper_phone'],
                mean_and_ci(tp)
            )),
        })

    # between‐participant
    b = load_between_lists(pid)
    if b:
        ba, tm, tp = b
        between_rows.append({
            'participant': pid,
            'n_tests': len(ba),
            **dict(zip(
                ['mean_balanced_accuracy','CI95_lower_ba','CI95_upper_ba'],
                mean_and_ci(ba)
            )),
            **dict(zip(
                ['mean_true_movie_rate','CI95_lower_movie','CI95_upper_movie'],
                mean_and_ci(tm)
            )),
            **dict(zip(
                ['mean_true_phone_rate','CI95_lower_phone','CI95_upper_phone'],
                mean_and_ci(tp)
            )),
        })

    # within‐temporal ordering
    wt = load_within_tempord(pid)
    if wt:
        within_tempord_data[pid] = wt

    # between‐temporal ordering
    bt = load_between_tempord(pid)
    if bt:
        between_tempord_data[pid] = bt

    # within‐participant occlusion
    oc = load_within_occlusion(pid)
    if oc is not None:
        within_occl_data[pid] = oc

# build DataFrames
within_df   = pd.DataFrame(within_rows).set_index('participant')
between_df  = pd.DataFrame(between_rows).set_index('participant')

# ── POOLED ACROSS PARTICIPANTS ─────────────────────────────────────────────────
def summarize_series(s):
    m, lo, hi = mean_and_ci(s)
    return pd.Series({'mean':m,'CI95_lower':lo,'CI95_upper':hi,'n_participants':s.size})

within_pooled = pd.DataFrame({
    'balanced_accuracy': summarize_series(within_df['mean_balanced_accuracy']),
    'true_movie_rate':   summarize_series(within_df['mean_true_movie_rate']),
    'true_phone_rate':   summarize_series(within_df['mean_true_phone_rate']),
}).T
# ensure integer count
within_pooled['n_participants'] = within_pooled['n_participants'].astype(int)

between_pooled = pd.DataFrame({
    'balanced_accuracy': summarize_series(between_df['mean_balanced_accuracy']),
    'true_movie_rate':   summarize_series(between_df['mean_true_movie_rate']),
    'true_phone_rate':   summarize_series(between_df['mean_true_phone_rate']),
}).T
# ensure integer count
between_pooled['n_participants'] = between_pooled['n_participants'].astype(int)

# ── POOLED WITHIN‐TEMPORAL ORDERING ─────────────────────────────────────────────
if within_tempord_data:
    dist_map = {}
    for ba_list, bd_list in within_tempord_data.values():
        for ba, d in zip(ba_list, bd_list):
            dist_map.setdefault(d, []).append(ba)
    within_tempord_pooled = pd.DataFrame([
        {
            **dict(zip(
                ['mean','CI95_lower','CI95_upper'],
                mean_and_ci(dist_map[d])
            )),
            'n_participants': len(dist_map[d])
        }
        for d in sorted(dist_map)
    ], index=[f"Distance {d}" for d in sorted(dist_map)])
    # ensure integer count
    within_tempord_pooled['n_participants'] = within_tempord_pooled['n_participants'].astype(int)

# ── POOLED BETWEEN‐TEMPORAL ORDERING ────────────────────────────────────────────
if between_tempord_data:
    dist_map_bt = {}
    for ba_means, bd_list in between_tempord_data.values():
        for ba, d in zip(ba_means, bd_list):
            dist_map_bt.setdefault(d, []).append(ba)
    between_tempord_pooled = pd.DataFrame([
        {
            **dict(zip(
                ['mean','CI95_lower','CI95_upper'],
                mean_and_ci(dist_map_bt[d])
            )),
            'n_participants': len(dist_map_bt[d])
        }
        for d in sorted(dist_map_bt)
    ], index=[f"Distance {d}" for d in sorted(dist_map_bt)])
    # ensure integer count
    between_tempord_pooled['n_participants'] = between_tempord_pooled['n_participants'].astype(int)

# ── POOLED OCCLUSION ANALYSIS ──────────────────────────────────────────────────
if within_occl_data:
    # collect per‐cluster across participants
    n_clust = len(next(iter(within_occl_data.values())))
    cl_map = {i: [] for i in range(n_clust)}
    for deltas in within_occl_data.values():
        for i, delta in enumerate(deltas):
            cl_map[i].append(delta)
    occl_pooled = pd.DataFrame([
        {
            **dict(zip(
                ['mean','CI95_lower','CI95_upper'],
                mean_and_ci(cl_map[i])
            )),
            'n_participants': len(cl_map[i])
        }
        for i in sorted(cl_map)
    ], index=[f"Cluster {i+1}" for i in sorted(cl_map)])
    # ensure integer count
    occl_pooled['n_participants'] = occl_pooled['n_participants'].astype(int)

# ── OUTPUT ──────────────────────────────────────────────────────────────────────
print("\nPooled within-participant results across participants:\n")
print(within_pooled)

print("\nPooled between-participant generalization across participants:\n")
print(between_pooled)

print("\nPooled within-participant temporal ordering across participants:\n")
print(within_tempord_pooled)

print("\nPooled between-participant temporal ordering across participants:\n")
print(between_tempord_pooled)

print("\nPooled occlusion analysis across participants:\n")
print(occl_pooled)