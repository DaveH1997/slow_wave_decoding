import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats

# Toggle saving figures
SAVE_FIGURES = False
# Toggle display of real IDs (False=generic labels)
DISPLAY_IDs = False

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR    = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Tap Classification/TC Output'
WITHIN_RUN  = 'within'
BETWEEN_RUN = 'between'
OUTPUT_DIR  = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Tap Classification/tc_plots'
METRICS     = ['balanced_accuracy', 'true_notap_rate', 'true_tap_rate']
# channel clusters for prototype boundaries
clusters = [
    [49,60,48,32,47,59], [34,19,33,20,7,18,8], [35,50,36,21,37,51],
    [17,31,16,46,30,15], [1,6,2,5,3,4], [9,22,10,38,23,11],
    [58,29,45,44,57,61], [14,12,13,28,25,27,26], [52,24,39,40,53,62],
    [43,41,42,56,54,55]
]
clusters = [[i-1 for i in cl] for cl in clusters]
channel_bounds = np.cumsum([len(cl) for cl in clusters])

# ── HELPERS ────────────────────────────────────────────────────────────────────
def mean_and_ci(arr, alpha=0.05):
    a = np.asarray(arr, dtype=float)
    m = a.mean()
    sem = stats.sem(a, ddof=1)
    t = stats.t.ppf(1 - alpha/2, df=a.size-1)
    return m, m - t * sem, m + t * sem

# Discover participants and lags
participants = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
)
if participants:
    sample = participants[0]
    lag_dirs = [d for d in os.listdir(os.path.join(BASE_DIR, sample)) if d.startswith('time_lag_')]
    LAGS = sorted(int(d.split('_')[-1]) for d in lag_dirs)
else:
    LAGS = []

# ── LOAD FUNCTIONS ─────────────────────────────────────────────────────────────
def load_fold(pid, lag):
    p = os.path.join(BASE_DIR, pid, f'time_lag_{lag}', WITHIN_RUN, 'results.pkl')
    if not os.path.isfile(p): return None
    return pickle.load(open(p,'rb'))['folds']

def load_between(pid, lag):
    p = os.path.join(BASE_DIR, pid, f'time_lag_{lag}', BETWEEN_RUN, 'results.pkl')
    if not os.path.isfile(p): return None
    return pickle.load(open(p,'rb'))['folds']

def load_prototypes(pid, lag):
    p = os.path.join(BASE_DIR, pid, f'time_lag_{lag}', WITHIN_RUN, 'results.pkl')
    if not os.path.isfile(p): return None
    overall = pickle.load(open(p,'rb')).get('overall', {})
    pm = overall.get('mean_proto_notap', None)
    if pm is None:
        pm = overall.get('mean_prototype_notap', None)
    pt = overall.get('mean_proto_tap', None)
    if pt is None:
        pt = overall.get('mean_prototype_tap', None)
    return pm, pt

# ── PLOTTING PER LAG ───────────────────────────────────────────────────────────
for lag in LAGS:
    # create output subdir only if saving
    out_sub = os.path.join(OUTPUT_DIR, f'time_lag_{lag}')
    if SAVE_FIGURES:
        os.makedirs(out_sub, exist_ok=True)

    # collect within/between
    within, between = {m:{} for m in METRICS}, {m:{} for m in METRICS}
    for pid in participants:
        folds = load_fold(pid, lag)
        if folds:
            for m in METRICS:
                within[m][pid] = [fd[m] for fd in folds]
        bfolds = load_between(pid, lag)
        if bfolds:
            for fd in bfolds:
                t = fd.get('test_participant') or fd.get('test_participant_ID')
                for m in METRICS:
                    between[m].setdefault(pid, {}).setdefault(t, []).append(fd[m])

    # sort participants by BA CI
    stats_dict = {pid: mean_and_ci(within['balanced_accuracy'][pid]) for pid in within['balanced_accuracy']}
    high = [p for p,(m,lo,hi) in stats_dict.items() if lo > 0.5]
    low  = [p for p in stats_dict if p not in high]
    high.sort(key=lambda p: stats_dict[p][0], reverse=True)
    low.sort(key=lambda p: stats_dict[p][0], reverse=True)
    sorted_pids = high + low
    N = len(sorted_pids)
    labels = sorted_pids if DISPLAY_IDs else [f'P{N-i:02d}' for i in range(N)]

    # build mean & full CI matrices
    mean_mat, ci_lo_mat, ci_hi_mat = {}, {}, {}
    for m in METRICS:
        mm  = np.full((N,N), np.nan)
        clo = np.full((N,N), np.nan)
        chi = np.full((N,N), np.nan)
        for i, tr in enumerate(sorted_pids):
            for j, te in enumerate(sorted_pids):
                vals = within[m][tr] if tr == te else between[m].get(tr, {}).get(te, [])
                if vals:
                    mv, lo, hi = mean_and_ci(vals)
                else:
                    mv, lo, hi = np.nan, np.nan, np.nan
                mm[i,j], clo[i,j], chi[i,j] = mv, lo, hi
        mean_mat[m]  = mm
        ci_lo_mat[m] = clo
        ci_hi_mat[m] = chi

    # create red–white–green display matrices
    disp = {}
    ba_lo = ci_lo_mat['balanced_accuracy']
    for m in METRICS:
        dm = np.full((N,N), 0.5)
        for i in range(N):
            for j in range(N):
                lo, hi = ci_lo_mat[m][i,j], ci_hi_mat[m][i,j]
                val     = mean_mat[m][i,j]
                # above chance
                if lo > 0.5:
                    dm[i,j] = np.clip(val, 0.5, 1.0)
                # below chance
                elif hi < 0.5:
                    dm[i,j] = np.clip(val, 0.0, 0.5)
                # else stay at 0.5 (white)
                # mask secondaries where BA is not significant
                if m != 'balanced_accuracy' and ba_lo[i,j] <= 0.5:
                    dm[i,j] = 0.5
        disp[m] = dm

    # MATRIX PLOT (unchanged)
    cmap = LinearSegmentedColormap.from_list('red_white_green', ['red','white','green'])
    fig = plt.figure(figsize=(18,6), dpi=300)
    gs = gridspec.GridSpec(1,4, width_ratios=[1,1,1,0.06], wspace=0.1)
    axes = [fig.add_subplot(gs[0,i]) for i in range(3)]; cax = fig.add_subplot(gs[0,3])
    titles = ['Balanced Accuracy','True NoTap Rate','True Tap Rate']
    for ax, m, title in zip(axes, METRICS, titles):
        im = ax.imshow(disp[m], cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        if ax is axes[0]:
            ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=6)
            ax.set_ylabel('Train Participant')
        else:
            ax.set_yticks([])
        ax.set_xlabel('Test Participant'); ax.set_title(title)
        ax.set_xticks(np.arange(-.5, N,1), minor=True)
        ax.set_yticks(np.arange(-.5, N,1), minor=True)
        ax.grid(which='minor', color='black', linewidth=0.3)
        ax.tick_params(which='minor', length=0)
        ax.invert_xaxis()
    cb = fig.colorbar(im, cax=cax); cb.set_label('Mean Accuracy')
    mp = axes[0].get_position(); cp = cax.get_position()
    cax.set_position([cp.x0, mp.y0, cp.width, mp.height])
    if SAVE_FIGURES:
        fig.savefig(os.path.join(out_sub, f'matrix_lag{lag}.svg'), bbox_inches='tight')
    plt.show(); plt.close(fig)

    # ── PROTOTYPE PLOTS ────────────────────────────────────────────────────────
    proto_nt, proto_t = [], []
    for pid in participants:
        pmpt = load_prototypes(pid, lag)
        if pmpt is None: continue
        pm, pt = pmpt
        if pm is None or pt is None: continue
        proto_nt.append(pm); proto_t.append(pt)
    if not proto_nt: continue
    arr_nt = np.stack(proto_nt); arr_t = np.stack(proto_t)
    mean_nt = arr_nt.mean(0); mean_t = arr_t.mean(0)
    allv = np.concatenate([mean_nt.ravel(), mean_t.ravel()])
    lo, hi = np.percentile(allv, [2.5,97.5])
    def rescale(x): return (np.clip(x, lo, hi) - lo) / (hi - lo)
    R_nt = rescale(mean_nt); R_t = rescale(mean_t)
    mu, sig = allv.mean(), allv.std(ddof=1)
    Z_nt = (mean_nt - mu) / sig; Z_t = (mean_t - mu) / sig
    zlim = max(abs(Z_nt).max(), abs(Z_t).max())

    fig = plt.figure(figsize=(12,10))
    left, bottom = 0.05, 0.07; space_h, space_v = 0.03, 0.08
    w = (1-left*2-space_h)/2; h = (1-bottom*2-space_v)/2; cbar_w = 0.02
    ax1 = fig.add_axes([left, bottom+h+space_v, w, h])
    ax2 = fig.add_axes([left+w+space_h, bottom+h+space_v, w, h], sharey=ax1)
    cax1= fig.add_axes([left+2*w+2*space_h, bottom+h+space_v, cbar_w, h])
    ax3 = fig.add_axes([left, bottom, w, h])
    ax4 = fig.add_axes([left+w+space_h, bottom, w, h], sharey=ax3)
    cax2= fig.add_axes([left+2*w+2*space_h, bottom, cbar_w, h])

    # rescaled white->black
    wb = LinearSegmentedColormap.from_list('white_black',['white','black'])
    im1 = ax1.imshow(R_nt, aspect='auto', vmin=0, vmax=1, cmap=wb)
    ax1.set_ylabel('Channel (cluster-ordered)')
    ax2.imshow(R_t, aspect='auto', vmin=0, vmax=1, cmap=wb)
    # cluster boundaries on both
    for b in channel_bounds[:-1]:
        ax1.axhline(b-0.5, color='green', linestyle='--', linewidth=1)
        ax2.axhline(b-0.5, color='green', linestyle='--', linewidth=1)

    # z-scored with boundaries
    im2 = ax3.imshow(Z_nt, aspect='auto', vmin=-zlim, vmax=zlim, cmap='bwr')
    ax3.set_xlabel('Time Bin'); ax3.set_ylabel('Channel (cluster-ordered)')
    im3 = ax4.imshow(Z_t, aspect='auto', vmin=-zlim, vmax=zlim, cmap='bwr')
    ax4.set_xlabel('Time Bin')
    for b in channel_bounds[:-1]:
        ax3.axhline(b-0.5, color='green', linestyle='--', linewidth=1)
        ax4.axhline(b-0.5, color='green', linestyle='--', linewidth=1)

    # remove ticks and set titles
    for ax, title in zip([ax1,ax2,ax3,ax4],
                         ['Rescaled NoTap Prototype','Rescaled Tap Prototype',
                          'Z-Scored NoTap Prototype','Z-Scored Tap Prototype']):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)

    # colorbars
    cb1 = fig.colorbar(im1, cax=cax1); cb1.set_label('Rescaled Activation')
    cb2 = fig.colorbar(im2, cax=cax2); cb2.set_label('Z-Score')
    if SAVE_FIGURES:
        fig.savefig(os.path.join(out_sub, f'prototypes_lag{lag}.svg'), bbox_inches='tight')
    plt.show(); plt.close(fig)
