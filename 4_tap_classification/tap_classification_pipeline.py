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
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BalancedAccuracy(Metric):
    """Balanced accuracy metric combining sensitivity and specificity."""
    def __init__(self, name='balanced_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', shape=(), initializer='zeros')
        self.tn = self.add_weight(name='tn', shape=(), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
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
        sens = self.tp / (self.tp + self.fn + K.epsilon())
        spec = self.tn / (self.tn + self.fp + K.epsilon())
        return (sens + spec) / 2.0

    def reset_state(self):
        for v in (self.tp, self.tn, self.fp, self.fn):
            v.assign(0.)

# Helper to apply time lag to segments & labels
def apply_time_lag(segments, labels, time_lag):
    """
    Shift labels by time_lag relative to segments.
      • time_lag > 0: drop first time_lag segments, last time_lag labels
      • time_lag < 0: drop last |time_lag| segments, first |time_lag| labels
      • time_lag = 0: no change
    """
    if time_lag > 0:
        segments = segments[time_lag:]
        labels   = labels[:-time_lag]
    elif time_lag < 0:
        lag = abs(time_lag)
        segments = segments[:-lag]
        labels   = labels[lag:]
    return segments, labels

# Load and preprocess data for tap classification
def load_data(pid, base_dir):
    seg_path = os.path.join(base_dir, f"{pid}_segments.npy")
    lbl_path = os.path.join(base_dir, f"{pid}_labels.npy")
    tap_path = os.path.join(base_dir, f"{pid}_tap_counts.npy")

    segments   = np.load(seg_path)
    cond_lbls  = np.load(lbl_path).ravel()  # 0=Movie, 1=Phone
    tap_counts = np.load(tap_path).ravel()

    phone_mask = (cond_lbls == 1)
    segments   = segments[phone_mask]
    taps       = tap_counts[phone_mask]
    labels_cls = (taps > 0).astype(int)

    # Exclude ocular channels and reorder by clusters
    segments = segments[:, :62, :]
    clusters = [
        [49,60,48,32,47,59], [34,19,33,20,7,18,8],
        [35,50,36,21,37,51], [17,31,16,46,30,15],
        [1,6,2,5,3,4], [9,22,10,38,23,11],
        [58,29,45,44,57,61], [14,12,13,28,25,27,26],
        [52,24,39,40,53,62], [43,41,42,56,54,55]
    ]
    clusters = [[i-1 for i in cl] for cl in clusters]
    order = [ch for cl in clusters for ch in cl]
    segments = segments[:, order, :]
    segments = segments[..., np.newaxis]

    return segments, labels_cls, clusters

def make_blocks(indices, n_folds=5, overlap=0.5):
    n = len(indices)
    base = n // n_folds
    drop = 1 if overlap <= 0.5 else int(np.ceil(1/(1-overlap)))
    height = base - drop
    idx = indices[: base * n_folds].reshape(base, n_folds)
    return idx[: height, :]

def build_model(input_shape):
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(4, (3,3), padding='same',
        kernel_regularizer=keras.regularizers.l2(1e-3))(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(8, (3,3), padding='same',
        kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu',
        kernel_regularizer=keras.regularizers.l2(5e-2))(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out)
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=[BalancedAccuracy()]
    )
    return model

def activation_maximization(model, target, iters=300, lr=0.01, l2_reg=1e-3):
    shape = (1,) + model.input_shape[1:]
    x = tf.Variable(tf.random.uniform(shape))
    sign = -1 if target == 1 else 1
    opt = keras.optimizers.Adam(lr)
    history = []
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x)
            loss = sign * pred + l2_reg * tf.reduce_sum(tf.square(x))
        grads = tape.gradient(loss, x)
        opt.apply_gradients([(grads, x)])
        x.assign(tf.clip_by_value(x, 0, 1))
        history.append(loss.numpy().item())
    return x.numpy().squeeze(), history

def run_cv(pid, base_dir, out_dir,
           occlusion=False, activation=False,
           between=False, others=None,
           time_lag=0):

    # Determine which lags to run
    lags = [0] if time_lag == 0 else list(range(-time_lag, time_lag+1))
    all_results = {}

    for lag in lags:
        mode = 'between' if between else 'within'
        print(f"Starting CV for {pid}: mode={mode}, time_lag={lag}")

        # Load data and apply lag if needed
        segs, labs, clusters = load_data(pid, base_dir)
        if lag != 0:
            segs, labs = apply_time_lag(segs, labs, lag)

        cluster_lens = [len(c) for c in clusters]
        occl_sets = []
        start = 0
        for l in cluster_lens:
            occl_sets.append(list(range(start, start+l)))
            start += l

        # Prepare CV folds
        blocks = make_blocks(np.arange(len(labs)))
        input_shape = segs.shape[1:]

        # Set up output directory for this lag
        run_dir = os.path.join(out_dir, pid, f"time_lag_{lag}", mode)
        os.makedirs(run_dir, exist_ok=True)

        results = {'folds': [], 'overall': {}}
        histories, all_true, all_pred = [], [], []
        occl_deltas, proto_notap, proto_tap = [], [], []
        am_loss_notap, am_loss_tap = [], []

        n_folds = blocks.shape[1]
        for fold in range(n_folds):
            print(f"Fold {fold+1}/{n_folds}")
            test_idx = blocks[:, fold]
            rem = blocks[:, np.arange(n_folds) != fold]
            val_idx = rem[:, np.random.choice(rem.shape[1])]
            train_idx = np.setdiff1d(rem.ravel(), val_idx)

            X_tr, y_tr = shuffle(segs[train_idx], labs[train_idx], random_state=42)
            X_val, y_val = segs[val_idx], labs[val_idx]
            cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
            cw_dict = dict(zip(np.unique(y_tr), cw))

            model = build_model(input_shape)
            es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            rl = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15)
            hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                             epochs=500, batch_size=64, verbose=0,
                             callbacks=[es, rl], class_weight=cw_dict)
            histories.append(hist.history)

            preds = (model.predict(segs[test_idx]) > 0.5).astype(int).ravel()
            ba = balanced_accuracy_score(labs[test_idx], preds)
            cm = confusion_matrix(labs[test_idx], preds, normalize='true')
            true_notap = cm[0, 0]
            true_tap   = cm[1, 1]
            all_true.extend(labs[test_idx]); all_pred.extend(preds)
            fold_res = {
                'fold': fold+1,
                'balanced_accuracy': ba,
                'true_notap_rate': float(true_notap),
                'true_tap_rate':   float(true_tap)
            }

            if occlusion:
                deltas = []
                for cl in occl_sets:
                    Xo = segs[test_idx].copy()
                    Xo[:, cl, :, :] = 0
                    po = (model.predict(Xo) > 0.5).astype(int).ravel()
                    ba_o = balanced_accuracy_score(labs[test_idx], po)
                    deltas.append(max(ba,0.5) - max(ba_o,0.5))
                fold_res['occlusion_deltas'] = deltas
                occl_deltas.append(deltas)

            if activation:
                xn, ln = activation_maximization(model, 0)
                xt, lt = activation_maximization(model, 1)
                fold_res.update({
                    'prototype_notap': xn, 'prototype_tap': xt,
                    'am_loss_notap': ln, 'am_loss_tap': lt
                })
                proto_notap.append(xn); proto_tap.append(xt)
                am_loss_notap.append(ln); am_loss_tap.append(lt)

            results['folds'].append(fold_res)

            if between and others:
                for tpid in others:
                    seg_o, lbl_o, _ = load_data(tpid, base_dir)
                    seg_o, lbl_o = apply_time_lag(seg_o, lbl_o, lag)
                    preds_o = (model.predict(seg_o) > 0.5).astype(int).ravel()
                    cm_o = confusion_matrix(lbl_o, preds_o, normalize='true')
                    ba_o = balanced_accuracy_score(lbl_o, preds_o)
                    results['folds'].append({
                        'fold': fold+1,
                        'test_participant': tpid,
                        'balanced_accuracy': float(ba_o),
                        'true_notap_rate': float(cm_o[0,0]),
                        'true_tap_rate':   float(cm_o[1,1])
                    })

        # Aggregate overall results
        if not between:
            folds = [f for f in results['folds'] if 'test_participant' not in f]
            bas      = [f['balanced_accuracy']   for f in folds]
            notap_rs = [f['true_notap_rate']     for f in folds]
            tap_rs   = [f['true_tap_rate']       for f in folds]
            results['overall'] = {
                'mean_balanced_accuracy': float(np.mean(bas)),
                'mean_true_notap_rate':   float(np.mean(notap_rs)),
                'mean_true_tap_rate':     float(np.mean(tap_rs))
            }
            if occlusion:
                results['overall']['mean_occlusion_deltas'] = np.mean(occl_deltas, axis=0).tolist()
            if activation:
                results['overall']['mean_proto_notap'] = np.mean(proto_notap, axis=0)
                results['overall']['mean_proto_tap']   = np.mean(proto_tap, axis=0)
        else:
            overall = {}
            for tpid in others or []:
                entries  = [e for e in results['folds'] if e.get('test_participant') == tpid]
                bas      = [e['balanced_accuracy'] for e in entries]
                notap_rs = [e['true_notap_rate']   for e in entries]
                tap_rs   = [e['true_tap_rate']     for e in entries]
                overall[tpid] = {
                    'mean_balanced_accuracy': float(np.mean(bas)),
                    'mean_true_notap_rate':   float(np.mean(notap_rs)),
                    'mean_true_tap_rate':     float(np.mean(tap_rs))
                }
            results['overall'] = overall

        # Save results
        with open(os.path.join(run_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        # Save plots (within-subject only)
        if not between:
            cm_all = confusion_matrix(all_true, all_pred, normalize='true')
            plt.figure(figsize=(4,3))
            sns.heatmap(cm_all, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=['NoTap','Tap'], yticklabels=['NoTap','Tap'])
            plt.title('Pooled Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"{pid}_pooled_conf.png"))
            plt.close()

            fig, axes = plt.subplots(nrows=n_folds, ncols=2, figsize=(12,4*n_folds), sharex=True)
            for i, h in enumerate(histories):
                axes[i,0].plot(h['loss'], label='Train Loss')
                axes[i,0].plot(h['val_loss'], label='Val Loss')
                axes[i,0].set_title(f'Fold {i+1} Loss')
                axes[i,0].legend()
                axes[i,1].plot(h['balanced_accuracy'], label='Train BA')
                axes[i,1].plot(h['val_balanced_accuracy'], label='Val BA')
                axes[i,1].set_title(f'Fold {i+1} Balanced Accuracy')
                axes[i,1].legend()
            axes[-1,0].set_xlabel('Epoch'); axes[-1,1].set_xlabel('Epoch')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"{pid}_train_history.png"))
            plt.close()

            if occlusion:
                cluster_labels = []
                delta_vals     = []
                for deltas in occl_deltas:
                    for idx, d in enumerate(deltas):
                        cluster_labels.append(f"Cluster {idx+1}")
                        delta_vals.append(d)
                plt.figure(figsize=(10,6))
                sns.barplot(
                    x=cluster_labels,
                    y=delta_vals,
                    color='#4C72B0',
                    errorbar=('ci',95),
                    capsize=0.2
                )
                plt.xticks(rotation=45)
                plt.xlabel('Cluster'); plt.ylabel('Δ BA')
                plt.title('Cluster Occlusion (Δ BA)')
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f"{pid}_occlusion.png"))
                plt.close()

            if activation:
                mean_n = np.mean(proto_notap, axis=0)
                mean_t = np.mean(proto_tap, axis=0)
                cum = np.cumsum([len(c) for c in clusters])
                plt.figure(figsize=(12,5))
                plt.subplot(1,2,1)
                plt.imshow(mean_n, aspect='auto', vmin=0, vmax=1, cmap='gray_r')
                for b in cum[:-1]: plt.axhline(b-0.5, color='green', linestyle='--')
                plt.title('Mean Prototype NoTap')
                plt.subplot(1,2,2)
                plt.imshow(mean_t, aspect='auto', vmin=0, vmax=1, cmap='gray_r')
                for b in cum[:-1]: plt.axhline(b-0.5, color='green', linestyle='--')
                plt.title('Mean Prototype Tap')
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f"{pid}_prototypes.png"))
                plt.close()

                plt.figure(figsize=(12,5))
                plt.subplot(1,2,1)
                for i, lh in enumerate(am_loss_notap):
                    plt.plot(lh, label=f'Fold {i+1}')
                plt.title('AM Loss — NoTap'); plt.xlabel('Iteration'); plt.ylabel('Loss')
                plt.legend()
                plt.subplot(1,2,2)
                for i, lh in enumerate(am_loss_tap):
                    plt.plot(lh, label=f'Fold {i+1}')
                plt.title('AM Loss — Tap'); plt.xlabel('Iteration'); plt.ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f"{pid}_am_loss_histories.png"))
                plt.close()

        all_results[lag] = results

    return all_results

if __name__ == '__main__':
    BASE = '/data1/s3821013/prepared_for_decoding/25_bins'
    OUT  = '/data1/s3821013/tap_classification_output'
    os.makedirs(OUT, exist_ok=True)
    start_participant = 1   # select participant index to start with (one-based, inclusive)
    end_participant = 46   # select participant index to end with (one-based, inclusive)
    all_participants = sorted({f[:4] for f in os.listdir(BASE) if 'segments' in f})
    participants_to_run = all_participants[start_participant-1:end_participant]
    for pid in participants_to_run:
        run_cv(pid, BASE, OUT, occlusion=True, activation=True, time_lag=6)
        run_cv(pid, BASE, OUT, between=True, others=[p for p in all_participants if p!=pid], time_lag=6)
