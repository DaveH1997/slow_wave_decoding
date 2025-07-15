import os
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns

# Toggle saving figures to disk
SAVE_FIGURES = False

# Toggle whether to display participant IDs or generic labels in the matrix plots
DISPLAY_IDs = False

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# BASE_DIR    = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Condition Classification/Python/CC Output/Raw'
BASE_DIR    = '/Users/davidhof/Desktop/MSc/5th Semester/Thesis/Condition Classification/Python/CC Output/Cleaned'
WITHIN_RUN  = 'within'
BETWEEN_RUN = 'between'
MAT_FIG     = 'condition_classification_matrix.svg'
TEMP_FIG    = 'temporal_ordering_summary.svg'
PROTO_FIG   = 'activation_prototypes_composite.svg'

# ── HELPERS ────────────────────────────────────────────────────────────────────
def mean_and_ci(arr, alpha=0.05):
    arr   = np.asarray(arr)
    mean  = arr.mean()
    sem   = stats.sem(arr, ddof=1)
    t_val = stats.t.ppf(1 - alpha/2, df=arr.size - 1)
    return mean, mean - t_val * sem, mean + t_val * sem

# ── LOAD RESULTS ───────────────────────────────────────────────────────────────
metrics = ['balanced_accuracy', 'true_movie_rate', 'true_phone_rate']
within, between = {m: {} for m in metrics}, {m: {} for m in metrics}

for pid in sorted(os.listdir(BASE_DIR)):
    # within-participant
    p_within = os.path.join(BASE_DIR, pid, WITHIN_RUN, 'results.pkl')
    if os.path.exists(p_within):
        with open(p_within, 'rb') as f:
            folds = pickle.load(f)['folds']
        for m in metrics:
            within[m][pid] = [fd[m] for fd in folds]
    # between-participant
    p_between = os.path.join(BASE_DIR, pid, BETWEEN_RUN, 'results.pkl')
    if os.path.exists(p_between):
        with open(p_between, 'rb') as f:
            folds = pickle.load(f)['folds']
        for m in metrics:
            between[m].setdefault(pid, {})
        for fd in folds:
            tpid = fd['test_participant_ID']
            for m in metrics:
                between[m][pid].setdefault(tpid, []).append(fd[m])

# ── SORT PARTICIPANTS (by within-BA with CI criteria) ──────────────────────────
pid_stats = {pid: mean_and_ci(within['balanced_accuracy'][pid])
             for pid in within['balanced_accuracy']}
group_high = [pid for pid, (_, ci_lo, _) in pid_stats.items() if ci_lo > 0.5]
group_low  = [pid for pid, (_, ci_lo, _) in pid_stats.items() if ci_lo <= 0.5]
group_high_sorted = sorted(group_high,
                            key=lambda pid: pid_stats[pid][0],
                            reverse=True)
group_low_sorted  = sorted(group_low,
                            key=lambda pid: pid_stats[pid][0],
                            reverse=True)
sorted_pids = group_high_sorted + group_low_sorted
N = len(sorted_pids)
label_map = {pid: f'P{N-idx:02d}' for idx, pid in enumerate(sorted_pids)}

# ── BUILD MEAN / CI MATRICES FOR EACH METRIC ───────────────────────────────────
mean_mats, ci_lo_mats, ci_hi_mats = {}, {}, {}
for m in metrics:
    mean_mat = np.empty((N, N))
    ci_lo_mat = np.empty((N, N))
    ci_hi_mat = np.empty((N, N))
    for i, p_tr in enumerate(sorted_pids):
        for j, p_te in enumerate(sorted_pids):
            vals = within[m][p_tr] if p_tr == p_te else between[m].get(p_tr, {}).get(p_te, [])
            if vals:
                mean_val, ci_lo, ci_hi = mean_and_ci(vals)
            else:
                mean_val = ci_lo = ci_hi = np.nan
            mean_mat[i, j] = mean_val
            ci_lo_mat[i, j] = ci_lo
            ci_hi_mat[i, j] = ci_hi
    mean_mats[m]    = mean_mat
    ci_lo_mats[m]   = ci_lo_mat
    ci_hi_mats[m]   = ci_hi_mat

# ── CREATE DISPLAY MATRICES (apply colouring rules) ────────────────────────────
disp_mats = {}
ba_ci_lo = ci_lo_mats['balanced_accuracy']
ba_ci_hi = ci_hi_mats['balanced_accuracy']
for m in metrics:
    disp = np.full((N, N), 0.5)
    for i in range(N):
        for j in range(N):
            lo = ci_lo_mats[m][i, j]
            hi = ci_hi_mats[m][i, j]
            val = mean_mats[m][i, j]
            # above-chance
            if lo > 0.5:
                disp[i, j] = np.clip(val, 0.5, 1.0)
            # below-chance
            elif hi < 0.5:
                disp[i, j] = np.clip(val, 0.0, 0.5)
            # non-significant remain 0.5
            # for non-BA metrics mask where BA not significant
            if m != 'balanced_accuracy' and ba_ci_lo[i, j] <= 0.5:
                disp[i, j] = 0.5
    disp_mats[m] = disp

# ── PLOT SUMMARY MATRIX ───────────────────────────────────────────────────────
titles = ['Balanced Accuracy', 'True Movie Rate', 'True Phone Rate']
# dual red→white→green colormap
cmap = LinearSegmentedColormap.from_list('red_white_green', ['red', 'white', 'green'])
fig = plt.figure(figsize=(18, 6), dpi=600)
gs  = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,0.07], wspace=0.10)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax  = fig.add_subplot(gs[0, 3])

if DISPLAY_IDs:
    x_labels = y_labels = sorted_pids
else:
    x_labels = [label_map[p] for p in sorted_pids]
    y_labels = x_labels

for ax, m, title in zip(axes, metrics, titles):
    im = ax.imshow(disp_mats[m], cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
    if ax is axes[0]:
        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(y_labels, fontsize=6)
        ax.set_ylabel('Train Participant')
    else:
        ax.set_yticks([])
    ax.set_xlabel('Test Participant')
    ax.set_title(title)
    ax.set_xticks(np.arange(-.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.25)
    ax.tick_params(which='minor', length=0)
    ax.invert_xaxis()

cb = fig.colorbar(im, cax=cax)
cb.set_label('Mean Accuracy')
mat_pos  = axes[0].get_position()
cbar_pos = cax.get_position()
cax.set_position([cbar_pos.x0, mat_pos.y0, cbar_pos.width, mat_pos.height])

if SAVE_FIGURES:
    plt.savefig(MAT_FIG, format='svg', bbox_inches='tight')
plt.show()
plt.close(fig)
if SAVE_FIGURES:
    print(f"Matrix plot saved to {MAT_FIG}")

# ── TEMPORAL ORDERING SUMMARY ─────────────────────────────────────────────────
# Gather within-participant temporal ordering
temp_within = {}
for pid in sorted(os.listdir(BASE_DIR)):
    fn = os.path.join(BASE_DIR, pid, 'within_temp_ord', 'results.pkl')
    if not os.path.exists(fn):
        continue
    with open(fn, 'rb') as f:
        folds = pickle.load(f)['folds']
    for fd in folds:
        d = fd['block_distance']
        ba = fd['balanced_accuracy']
        temp_within.setdefault(d, []).append(ba)
dist_w = sorted(temp_within)
mean_w, lo_w, hi_w = [], [], []
for d in dist_w:
    m, lo, hi = mean_and_ci(temp_within[d])
    mean_w.append(m)
    lo_w.append(lo)
    hi_w.append(hi)

# Gather between-participant temporal ordering
temp_between = {}
for pid in sorted(os.listdir(BASE_DIR)):
    fn = os.path.join(BASE_DIR, pid, BETWEEN_RUN, 'results.pkl')
    if not os.path.exists(fn):
        continue
    with open(fn, 'rb') as f:
        folds = pickle.load(f)['folds']
    for fd in folds:
        if 'balanced_accuracy_temp_ord' not in fd:
            continue
        for d, ba in zip(fd['block_distance'], fd['balanced_accuracy_temp_ord']):
            temp_between.setdefault(d, []).append(ba)
dist_b = sorted(temp_between)
mean_b, lo_b, hi_b = [], [], []
for d in dist_b:
    m, lo, hi = mean_and_ci(temp_between[d])
    mean_b.append(m)
    lo_b.append(lo)
    hi_b.append(hi)

# Plot temporal ordering panels
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
# within-participant
sns.barplot(
    x=dist_w, y=mean_w, ax=axes[0],
    palette=sns.color_palette("viridis", len(dist_w)), dodge=False,
    ci=None
)
axes[0].errorbar(
    x=np.arange(len(dist_w)), y=mean_w,
    yerr=[np.array(mean_w)-np.array(lo_w), np.array(hi_w)-np.array(mean_w)],
    fmt='none', capsize=10, ecolor='black', elinewidth=1.75, capthick=1.75
)
axes[0].axhspan(0, 0.5, color='white', alpha=0.6)
axes[0].axhline(0.5, color='red', linestyle='--', lw=1.75)
axes[0].set_title('Within-Participant Temporal Ordering')
axes[0].set_xlabel('Block Distance')
axes[0].set_ylabel('Mean Balanced Accuracy')
axes[0].set_ylim(0, 1)

# between-participant
sns.barplot(
    x=dist_b, y=mean_b, ax=axes[1],
    palette=sns.color_palette("viridis", len(dist_b)), dodge=False,
    ci=None
)
axes[1].errorbar(
    x=np.arange(len(dist_b)), y=mean_b,
    yerr=[np.array(mean_b)-np.array(lo_b), np.array(hi_b)-np.array(mean_b)],
    fmt='none', capsize=10, ecolor='black', elinewidth=1.75, capthick=1.75
)
axes[1].axhspan(0, 0.5, color='white', alpha=0.6)
axes[1].axhline(0.5, color='red', linestyle='--', lw=1.75)
axes[1].set_title('Between-Participant Temporal Ordering')
axes[1].set_xlabel('Block Distance')
axes[1].set_yticks([])
axes[1].set_ylim(0, 1)

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(TEMP_FIG, format='svg', bbox_inches='tight')
plt.show()
plt.close(fig)
if SAVE_FIGURES:
    print(f"Temporal ordering plot saved to {TEMP_FIG}")

# ── COMBINED: ROBUST MIN–MAX (2.5–97.5%) & Z-SCORING ─────────────────────────
# cluster defs (for the green boundary lines)
clusters = [
    [49,60,48,32,47,59], [34,19,33,20,7,18,8], [35,50,36,21,37,51],
    [17,31,16,46,30,15], [1,6,2,5,3,4], [9,22,10,38,23,11],
    [58,29,45,44,57,61], [14,12,13,28,25,27,26], [52,24,39,40,53,62],
    [43,41,42,56,54,55]
]
clusters = [[i-1 for i in cl] for cl in clusters]
channel_bounds = np.cumsum([len(cl) for cl in clusters])

# load & mean-pool across participants
proto_m, proto_p = [], []
for pid in sorted(os.listdir(BASE_DIR)):
    fn = os.path.join(BASE_DIR, pid, WITHIN_RUN, 'results.pkl')
    if not os.path.exists(fn):
        continue
    with open(fn, 'rb') as f:
        res = pickle.load(f)
    o = res.get('overall', {})
    pm = o.get('mean_prototype_movie');  pp = o.get('mean_prototype_phone')
    if pm is not None and pp is not None:
        proto_m.append(pm);  proto_p.append(pp)

movie_arr = np.stack(proto_m, axis=0)
phone_arr = np.stack(proto_p, axis=0)
mean_movie = movie_arr.mean(axis=0)
mean_phone = phone_arr.mean(axis=0)

# robust 2.5–97.5% clip & scale
all_vals = np.concatenate([mean_movie.ravel(), mean_phone.ravel()])
p_low, p_high = np.percentile(all_vals, [2.5, 97.5])
def robust_rescale(mat, lo, hi):
    clipped = np.clip(mat, lo, hi)
    return (clipped - lo) / (hi - lo)
res_movie = robust_rescale(mean_movie, p_low, p_high)
res_phone = robust_rescale(mean_phone, p_low, p_high)

# global z-scoring
mu, sigma = all_vals.mean(), all_vals.std(ddof=1)
z_movie = (mean_movie - mu) / sigma
z_phone = (mean_phone - mu) / sigma
z_lim = max(np.abs(z_movie).max(), np.abs(z_phone).max())

# figure layout
fig = plt.figure(figsize=(12, 10))
left, bottom = 0.05, 0.07
space_h, space_v = 0.03, 0.08
width = (1 - left*2 - space_h) / 2
height = (1 - bottom*2 - space_v) / 2
cbar_w = 0.02

ax1 = fig.add_axes([left, bottom+height+space_v, width, height])
ax2 = fig.add_axes([left+width+space_h, bottom+height+space_v, width, height], sharey=ax1)
cax1 = fig.add_axes([left+2*width+2*space_h, bottom+height+space_v, cbar_w, height])
ax3 = fig.add_axes([left, bottom, width, height])
ax4 = fig.add_axes([left+width+space_h, bottom, width, height], sharey=ax3)
cax2 = fig.add_axes([left+2*width+2*space_h, bottom, cbar_w, height])

for ax, data, title in [
    (ax1, res_movie, 'Rescaled Movie Prototype'),
    (ax2, res_phone,'Rescaled Phone Prototype'),
]:
    im1 = ax.imshow(data, aspect='auto', vmin=0, vmax=1, cmap='gray_r')
    for b in channel_bounds[:-1]:
        ax.axhline(b-0.5, color='green', linestyle='--', linewidth=1)
    ax.set_title(title); ax.set_xticks([])
ax1.set_ylabel('Channel (cluster-ordered)')
ax2.set_yticks([])

for ax, data, title in [
    (ax3, z_movie, 'Z-scored Movie Prototype'),
    (ax4, z_phone,'Z-scored Phone Prototype'),
]:
    im2 = ax.imshow(data, aspect='auto', vmin=-z_lim, vmax=z_lim, cmap='bwr')
    for b in channel_bounds[:-1]:
        ax.axhline(b-0.5, color='green', linestyle='--', linewidth=1)
    ax.set_title(title); ax.set_xticks([])
ax3.set_xlabel('Time Bin'); ax3.set_ylabel('Channel (cluster-ordered)')
ax4.set_xlabel('Time Bin'); ax4.set_yticks([])

cb1 = fig.colorbar(im1, cax=cax1)
cb1.set_label('Rescaled Activation')
cb2 = fig.colorbar(im2, cax=cax2)
cb2.set_label('Z-score')

if SAVE_FIGURES:
    plt.savefig(PROTO_FIG, format='svg', bbox_inches='tight')
plt.show()
if SAVE_FIGURES:
    print(f"Composite figure saved to {PROTO_FIG}")