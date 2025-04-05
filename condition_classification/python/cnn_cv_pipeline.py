import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import pickle

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

# Fix randomness
np.random.seed(42)
tf.random.set_seed(42)

def load_data(participant_ID):
    """
    Load the pre‑exported .npy files for the given participant_ID and reshape the segments.
    Returns:
        segments (numpy.ndarray): Data reshaped to (N, 64, 100, 1).
        labels (numpy.ndarray): 1D label array.
    """
    base_dir = '/data1/s3821013/cnn_data'
    segments = np.load(os.path.join(base_dir, f"{participant_ID}_segments.npy"))
    labels = np.load(os.path.join(base_dir, f"{participant_ID}_labels.npy")).ravel()
    segments = segments[..., np.newaxis]  # Reshape for CNN: (N, 64, 100, 1)
    print(f"Participant {participant_ID}: Data shape:", segments.shape, "Labels shape:", labels.shape)
    return segments, labels

def make_blocks(class_indices, n_folds=5, overlap=0.5):
    """
    Utility function to split indices into blocks for cross-validation.
    """
    n = len(class_indices)
    base_size = n // n_folds
    drop = 1 if overlap <= 0.5 else int(np.ceil(1/(1-overlap)))
    block_height = base_size - drop
    if block_height < 1:
        raise ValueError("Overlap too large")
    idx = class_indices[:base_size * n_folds].reshape(base_size, n_folds)
    return idx[:block_height, :]

def build_model(input_shape):
    """
    Build and compile the CNN model.
    Args:
        input_shape (tuple): The shape of one input sample.
    Returns:
        model (keras.Model): The compiled CNN model.
    """
    inp = keras.Input(shape=input_shape)

    # Block 1
    x = keras.layers.Conv2D(4, (3,3), padding='same',
                             kernel_regularizer=keras.regularizers.l2(1e-3))(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    # Block 2
    x = keras.layers.Conv2D(8, (3,3), padding='same',
                             kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    # Flatten + Dense
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(8, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(1e-2))(x)
    x = keras.layers.Dropout(0.5)(x)

    # Output
    out = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

def run_cross_validation(segments, labels):
    """
    Perform 5-fold cross-validation on the data and return the results.
    Args:
        segments (numpy.ndarray): Input data.
        labels (numpy.ndarray): Labels corresponding to the data.
    Returns:
        results (dict): Dictionary containing fold-wise and overall metrics.
    """
    # Separate indices for each class
    movie_idx = np.where(labels == 0)[0]
    phone_idx = np.where(labels == 1)[0]

    movie_blocks = make_blocks(movie_idx)
    phone_blocks = make_blocks(phone_idx)

    # Randomize fold order
    movie_order = np.random.permutation(5)
    phone_order = np.random.permutation(5)

    scores = []
    all_y_true = []
    all_y_pred = []
    fold_results = []
    
    input_shape = segments.shape[1:]  # (64, 100, 1)

    # Perform 5‑fold cross‑validation
    for fold in range(5):
        # Select test blocks for each class
        test_movie = movie_blocks[:, movie_order[fold]]
        test_phone = phone_blocks[:, phone_order[fold]]

        # Select remaining blocks for train+val
        remaining_movie = movie_blocks[:, movie_order[np.arange(5) != fold]]
        remaining_phone = phone_blocks[:, phone_order[np.arange(5) != fold]]

        # Randomly choose one block from the remaining four for validation
        val_idx_movie = remaining_movie[:, np.random.choice(remaining_movie.shape[1])]
        val_idx_phone = remaining_phone[:, np.random.choice(remaining_phone.shape[1])]

        # Build train indices by excluding the validation block
        train_movie_idx = np.setdiff1d(remaining_movie.ravel(), val_idx_movie)
        train_phone_idx = np.setdiff1d(remaining_phone.ravel(), val_idx_phone)

        # Combine movie + phone indices for train/val/test
        train_idx = np.concatenate([train_movie_idx, train_phone_idx])
        val_idx = np.concatenate([val_idx_movie, val_idx_phone])
        test_idx = np.concatenate([test_movie, test_phone])

        # Extract arrays for this fold
        X_train, y_train = segments[train_idx], labels[train_idx]
        X_val, y_val = segments[val_idx], labels[val_idx]
        X_test, y_test = segments[test_idx], labels[test_idx]

        # Shuffle training set for randomness
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Compute balanced class weights
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, cw))

        # Build and compile a fresh CNN
        model = build_model(input_shape)
        earlystop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15, verbose=0
        )

        # Train the model
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=500,
            batch_size=64,
            verbose=0,
            callbacks=[earlystop, reduce_lr],
            class_weight=class_weights
        )

        # Evaluate on held‑out test fold
        preds = (model.predict(X_test) > 0.5).astype(int).ravel()
        bal_acc = balanced_accuracy_score(y_test, preds)
        print(f"Fold {fold+1} balanced accuracy: {bal_acc:.4f}")
        scores.append(bal_acc)

        # Save fold's confusion matrix and balanced accuracy
        cm = confusion_matrix(y_test, preds, normalize='true')
        fold_results.append({
            "fold": fold + 1,
            "balanced_accuracy": bal_acc,
            "confusion_matrix": cm
        })

        # Accumulate labels/predictions for pooled confusion matrix
        all_y_true.extend(y_test)
        all_y_pred.extend(preds)

    print("Mean balanced accuracy:", np.mean(scores))
    # Pooled confusion matrix across all folds
    cm_overall = confusion_matrix(all_y_true, all_y_pred, normalize='true')

    results = {
        "folds": fold_results,
        "overall": {
            "mean_balanced_accuracy": np.mean(scores),
            "confusion_matrix": cm_overall
        }
    }
    print("Results:", results)
    return results

def main():
    """
    Main function: find participant IDs from files with 'segments', run cross-validation for each, and save the results.
    """
    base_dir = '/data1/s3821013'
    # List all .npy files in the cnn_data directory that include 'segments'
    files = os.listdir(os.path.join(base_dir, 'cnn_data'))
    npy_files = [f for f in files if f.endswith('.npy') and 'segments' in f]
    # Extract participant IDs (first 7 characters of the file names) and sort them alphanumerically
    participant_ids = sorted([f[:7] for f in npy_files])
    
    all_results = {}
    for pid in participant_ids:
        print(f"Processing participant: {pid}")
        segments, labels = load_data(pid)
        results = run_cross_validation(segments, labels)
        all_results[pid] = results
    
    # Save all_results to file in base_dir
    results_path = os.path.join(base_dir, 'all_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved all results to {results_path}")

if __name__ == '__main__':
    main()