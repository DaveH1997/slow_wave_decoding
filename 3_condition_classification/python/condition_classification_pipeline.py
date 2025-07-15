import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Disable Grappler’s layout optimizer to suppress warnings
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BalancedAccuracy(Metric):
    # Balanced accuracy metric combining sensitivity and specificity
    def __init__(self, name='balanced_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', shape=(), initializer='zeros')
        self.tn = self.add_weight(name='tn', shape=(), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update counts from confusion matrix for the batch
        y_pred = tf.squeeze(tf.cast(y_pred > 0.5, tf.int32), axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2, dtype=self.dtype)
        self.tn.assign_add(cm[0, 0])
        self.fp.assign_add(cm[0, 1])
        self.fn.assign_add(cm[1, 0])
        self.tp.assign_add(cm[1, 1])

    def result(self):
        # Compute balanced accuracy = (sensitivity + specificity) / 2
        sens = self.tp / (self.tp + self.fn + K.epsilon())
        spec = self.tn / (self.tn + self.fp + K.epsilon())
        return (sens + spec) / 2.0

    def reset_state(self):
        # Reset metric state variables
        for v in (self.tp, self.tn, self.fp, self.fn):
            v.assign(0.)

# Load and preprocess data for participant
# Exclude ocular channels and reorder channels by clusters
def load_data(pid, base_dir):
    segments = np.load(os.path.join(base_dir, f"{pid}_segments.npy"))
    labels = np.load(os.path.join(base_dir, f"{pid}_labels.npy")).ravel()
    segments = segments[:, :62, :]
    clusters = [
        [49,60,48,32,47,59], [34,19,33,20,7,18,8], [35,50,36,21,37,51],
        [17,31,16,46,30,15], [1,6,2,5,3,4], [9,22,10,38,23,11],
        [58,29,45,44,57,61], [14,12,13,28,25,27,26], [52,24,39,40,53,62], [43,41,42,56,54,55]
    ]
    clusters = [[i-1 for i in cl] for cl in clusters]
    order = [ch for cl in clusters for ch in cl]
    segments = segments[:, order, :]
    segments = segments[..., np.newaxis]
    return segments, labels, clusters

# Create blocks for cross-validation
# Discard hinge segments according to overlap
def make_blocks(indices, n_folds=5, overlap=0.5):
    n = len(indices)
    base = n // n_folds
    drop = 1 if overlap <= 0.5 else int(np.ceil(1/(1-overlap)))
    h = base - drop
    idx = indices[: base * n_folds].reshape(base, n_folds)
    return idx[: h, :]

# Build CNN model
def build_model(input_shape):
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(4, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-3))(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(8, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-2))(x)
    x = keras.layers.Dropout(0.5)(x)

    out = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=[BalancedAccuracy()]
    )
    return model

# Generate class-specific prototype via activation maximization
def activation_maximization(model, target, iters=300, lr=0.01, l2_reg=1e-3):
    shape = (1,) + model.input_shape[1:]
    x = tf.Variable(tf.random.uniform(shape))
    sign = -1 if target == 1 else 1
    opt = keras.optimizers.Adam(lr)
    loss_history = []
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x)
            loss = sign * pred + l2_reg * tf.reduce_sum(x**2)
        grads = tape.gradient(loss, x)
        opt.apply_gradients([(grads, x)])
        x.assign(tf.clip_by_value(x, 0, 1))
        loss_history.append(loss.numpy().item())
    return x.numpy().squeeze(), loss_history

# Run cross-validation for a participant in specified mode
# Saves results.pkl and figures in structured subdirectories
def run_cross_validation(pid, base_dir, out_dir, temporal_order=False, occlusion=False, activation=False, between=False, others=None):
    run_name = 'between' if between else ('within_temp_ord' if temporal_order else 'within')
    run_dir = os.path.join(out_dir, pid, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting CV for {pid}: mode={run_name}")

    segs, labs, clusters = load_data(pid, base_dir)
    m_blk = make_blocks(np.where(labs == 0)[0])
    p_blk = make_blocks(np.where(labs == 1)[0])
    input_shape = segs.shape[1:]

    cluster_sizes = [len(c) for c in clusters]
    occl_clusters = []
    start = 0
    for size in cluster_sizes:
        occl_clusters.append(list(range(start, start + size)))
        start += size

    results = {'folds': [], 'overall': {}}
    histories, all_true, all_pred = [], [], []
    occl_accs, proto_m, proto_p = [], [], []
    am_loss_movie, am_loss_phone = [], []

    # Randomize block order
    om = np.random.permutation(5)
    op = np.random.permutation(5)

    for fold in range(5):
        print(f"Fold {fold+1}/5")

        # Select test indices
        if not temporal_order:
            tm = m_blk[:, om[fold]]
            tp = p_blk[:, op[fold]]
            keep_m = np.setdiff1d(np.arange(5), om[fold])
            keep_p = np.setdiff1d(np.arange(5), op[fold])
        else:
            tm = m_blk[:, fold]
            tp = p_blk[:, 4 - fold]
            keep_m = np.delete(np.arange(5), fold)
            keep_p = np.delete(np.arange(5), 4 - fold)
        test_idx = np.concatenate([tm, tp])
        rm = m_blk[:, keep_m]
        rp = p_blk[:, keep_p]

        # Select validation indices from remaining blocks
        vm = rm[:, np.random.choice(rm.shape[1])]
        vp = rp[:, np.random.choice(rp.shape[1])]
        val_idx = np.concatenate([vm, vp])

        # Prepare training indices (other blocks)
        tr_idx = np.concatenate([
            np.setdiff1d(rm.ravel(), vm),
            np.setdiff1d(rp.ravel(), vp)
        ])

        # Shuffle and compute class weights
        X_tr, y_tr = shuffle(segs[tr_idx], labs[tr_idx], random_state=42)
        X_val, y_val = segs[val_idx], labs[val_idx]
        cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        cw_dict = dict(zip(np.unique(y_tr), cw))

        # Train model
        model = build_model(input_shape)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        rl = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=0)
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=500, batch_size=64, verbose=0,
            callbacks=[es, rl], class_weight=cw_dict
        )
        histories.append(hist.history)

        # Evaluate on test set
        if not between:
            X_te, y_te = segs[test_idx], labs[test_idx]

            preds = (model.predict(X_te) > 0.5).astype(int).ravel()
            all_true.extend(y_te)
            all_pred.extend(preds)
            cm = confusion_matrix(y_te, preds, normalize='true')
            ba = balanced_accuracy_score(y_te, preds)

            fold_res = {
                'fold': fold+1,
                'balanced_accuracy': ba,
                'true_movie_rate': cm[0,0],
                'true_phone_rate': cm[1,1]
            }

            if temporal_order:
                dist = (5 - (fold + 1)) + ((4 - fold) + 1)
                fold_res['block_distance'] = dist

            if occlusion and not temporal_order:
                deltas = []
                for cl in occl_clusters:
                    Xo = X_te.copy()
                    Xo[:, cl, :, :] = 0
                    po = (model.predict(Xo) > 0.5).astype(int).ravel()
                    ba_o = balanced_accuracy_score(y_te, po)
                    deltas.append(max(ba,0.5) - max(ba_o,0.5))
                fold_res['occlusion_deltas'] = deltas
                occl_accs.append(deltas)

            if activation and not temporal_order:
                pm, loss_mov = activation_maximization(model, 0)
                pp, loss_phn = activation_maximization(model, 1)
                fold_res['prototype_movie']      = pm
                fold_res['prototype_phone']      = pp
                fold_res['am_loss_movie']        = loss_mov
                fold_res['am_loss_phone']        = loss_phn
                proto_m.append(pm)
                proto_p.append(pp)
                am_loss_movie.append(loss_mov)
                am_loss_phone.append(loss_phn)

            results['folds'].append(fold_res)
        else:
            # Between-participant evaluation on other participants
            for tpid in others or []:
                seg_tp, lab_tp, _ = load_data(tpid, base_dir)
                preds_full = (model.predict(seg_tp) > 0.5).astype(int).ravel()
                cm_f = confusion_matrix(lab_tp, preds_full, normalize='true')
                ba_f = balanced_accuracy_score(lab_tp, preds_full)
                fr = {
                    'fold': fold+1,
                    'test_participant_ID': tpid,
                    'balanced_accuracy': ba_f,
                    'true_movie_rate': cm_f[0,0],
                    'true_phone_rate': cm_f[1,1]
                }

                if temporal_order:
                    m_tp = make_blocks(np.where(lab_tp==0)[0])
                    p_tp = make_blocks(np.where(lab_tp==1)[0])
                    ba_l, tm_l, tp_l, dist_l = [], [], [], []
                    for i in range(5):
                        idx_s = np.concatenate([m_tp[:,i], p_tp[:,4-i]])
                        ys = lab_tp[idx_s]
                        ps = (model.predict(seg_tp[idx_s]) > 0.5).astype(int).ravel()
                        cm_s = confusion_matrix(ys, ps, normalize='true')
                        ba_s = balanced_accuracy_score(ys, ps)
                        dist_val = (5-(i+1)) + ((4-i)+1)
                        ba_l.append(ba_s)
                        tm_l.append(cm_s[0,0])
                        tp_l.append(cm_s[1,1])
                        dist_l.append(dist_val)
                    fr['block_distance'] = dist_l
                    fr['balanced_accuracy_temp_ord'] = ba_l
                    fr['true_movie_rate_temp_ord'] = tm_l
                    fr['true_phone_rate_temp_ord'] = tp_l

                results['folds'].append(fr)

    # Aggregate overall results
    if not between:
        bas = [e['balanced_accuracy'] for e in results['folds']]
        mts = [e['true_movie_rate'] for e in results['folds']]
        pts = [e['true_phone_rate'] for e in results['folds']]
        results['overall'] = {
            'mean_balanced_accuracy': float(np.mean(bas)),
            'mean_true_movie_rate': float(np.mean(mts)),
            'mean_true_phone_rate': float(np.mean(pts))
        }
        if occlusion and not temporal_order:
            mean_occl = np.mean(occl_accs, axis=0)
            results['overall']['mean_occlusion_deltas'] = mean_occl.tolist()
        if activation and not temporal_order:
            results['overall']['mean_prototype_movie'] = np.mean(proto_m, axis=0)
            results['overall']['mean_prototype_phone'] = np.mean(proto_p, axis=0)
    else:
        overall = {}
        for tpid in others or []:
            entries = [e for e in results['folds'] if e.get('test_participant_ID') == tpid]
            bas = [e['balanced_accuracy'] for e in entries]
            mts = [e['true_movie_rate'] for e in entries]
            pts = [e['true_phone_rate'] for e in entries]
            stats = {
                'mean_balanced_accuracy': float(np.mean(bas)),
                'mean_true_movie_rate': float(np.mean(mts)),
                'mean_true_phone_rate': float(np.mean(pts))
            }
            if temporal_order:
                stats['block_distance'] = entries[0]['block_distance']
                ba_mat = np.array([e['balanced_accuracy_temp_ord'] for e in entries])
                tm_mat = np.array([e['true_movie_rate_temp_ord'] for e in entries])
                tp_mat = np.array([e['true_phone_rate_temp_ord'] for e in entries])
                stats['mean_balanced_accuracy_temp_ord'] = ba_mat.mean(axis=0).tolist()
                stats['mean_true_movie_rate_temp_ord'] = tm_mat.mean(axis=0).tolist()
                stats['mean_true_phone_rate_temp_ord'] = tp_mat.mean(axis=0).tolist()
            overall[tpid] = stats
        results['overall'] = overall

    # Save results to disk
    with open(os.path.join(run_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Save figures to disk
    if not between and not temporal_order:
        # Pooled confusion matrix
        cm_all = confusion_matrix(all_true, all_pred, normalize='true')
        plt.figure(figsize=(4,3))
        sns.heatmap(cm_all, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Movie','Phone'], yticklabels=['Movie','Phone'])
        plt.title('Pooled Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{pid}_pooled_conf.png"))
        plt.close()

        # Training history grid
        fig, axes = plt.subplots(5,2,figsize=(12,20), sharex=True)
        for i, h in enumerate(histories):
            axes[i,0].plot(h['loss'],     label='Train Loss')
            axes[i,0].plot(h['val_loss'], label='Val Loss')
            axes[i,0].set_title(f'Fold {i+1} Loss')
            axes[i,0].legend()
            axes[i,1].plot(h['balanced_accuracy'],     label='Train BA')
            axes[i,1].plot(h['val_balanced_accuracy'], label='Val BA')
            axes[i,1].set_title(f'Fold {i+1} Balanced Accuracy')
            axes[i,1].legend()
        axes[-1,0].set_xlabel('Epoch')
        axes[-1,1].set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{pid}_train_history.png"))
        plt.close()

    if temporal_order and not between:
        # Block distance vs. balanced accuracy barplot
        bd      = [e['block_distance']    for e in results['folds']]
        ba_vals = [e['balanced_accuracy'] for e in results['folds']]
        order     = np.argsort(bd)
        bd_sorted = [bd[i]      for i in order]
        ba_sorted = [ba_vals[i] for i in order]
        palette = sns.color_palette('viridis_r', len(bd_sorted))
        plt.figure(figsize=(8,6))
        ax = sns.barplot(
            x=bd_sorted,
            y=ba_sorted,
            hue=bd,
            palette=palette,
            dodge=False
        )
        ax.legend_.remove()
        ax.set_xticks(range(len(bd_sorted)))
        ax.set_xticklabels(bd_sorted)
        ax.axhspan(0, 0.5, color='white', alpha=0.6)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.75)
        ax.set_ylim(0,1)
        ax.set_xlabel('Block Distance')
        ax.set_ylabel('Balanced Accuracy')
        plt.title('Temporal Ordering: BA vs. Block Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{pid}_temporal_ord.png"))
        plt.close()

    if occlusion and not temporal_order and not between:
        # Occlusion analysis barplot
        cluster_labels = []
        delta_vals    = []
        for deltas in occl_accs:
            for idx, delta in enumerate(deltas):
                cluster_labels.append(f"Cluster {idx+1}")
                delta_vals.append(delta)
        plt.figure(figsize=(10,6))
        sns.barplot(x=cluster_labels, y=delta_vals, color='#4C72B0', errorbar=('ci',95), capsize=0.2)
        plt.xticks(rotation=45)
        plt.xlabel('Cluster')
        plt.ylabel('Δ BA')
        plt.title('Cluster Occlusion (Δ BA)')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{pid}_occlusion.png"))
        plt.close()

    if activation and not temporal_order and not between:
        # Activation maximization prototypes
        mean_pm = np.mean(proto_m, axis=0)
        mean_pp = np.mean(proto_p, axis=0)
        cum = np.cumsum([len(cl) for cl in clusters])
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.imshow(mean_pm, aspect='auto', vmin=0, vmax=1, cmap='gray_r')
        for b in cum[:-1]: plt.axhline(b-0.5, color='green', linestyle='--')
        plt.title('Mean Prototype Movie')
        plt.subplot(1,2,2)
        plt.imshow(mean_pp, aspect='auto', vmin=0, vmax=1, cmap='gray_r')
        for b in cum[:-1]: plt.axhline(b-0.5, color='green', linestyle='--')
        plt.title('Mean Prototype Phone')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{pid}_prototypes.png"))
        plt.close()
        # Activation maximization loss histories
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for i, lh in enumerate(am_loss_movie):
            plt.plot(lh, label=f'Fold {i+1}')
        plt.title('AM Loss — Movie'); plt.xlabel('Iteration'); plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1,2,2)
        for i, lh in enumerate(am_loss_phone):
            plt.plot(lh, label=f'Fold {i+1}')
        plt.title('AM Loss — Phone'); plt.xlabel('Iteration'); plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{pid}_am_loss_histories.png"))
        plt.close()

    return results

if __name__=='__main__':
    BASE = '/data1/s3821013/prepared_for_decoding/100_bins'
    OUT = '/data1/s3821013/condition_classification_output'
    os.makedirs(OUT, exist_ok=True)
    all_participants = sorted({f[:4] for f in os.listdir(BASE) if 'segments' in f})
    participants_to_run = all_participants   # select participants
    for pid in participants_to_run:
        run_cross_validation(pid, BASE, OUT, temporal_order=False, occlusion=True, activation=True)
        run_cross_validation(pid, BASE, OUT, temporal_order=True)
        others = [p for p in all_participants if p!=pid]
        run_cross_validation(pid, BASE, OUT, temporal_order=True, between=True, others=others)
