# config.py
import os
import numpy as np
import scipy.io
import random
import lmdb
import pickle
from scipy.signal import butter, lfilter, resample

SEED = 42
FS = 250
LOWCUT = 0.01
HIGHCUT = 38
RESAMPLE_POINTS = 800
CROP_START = 2  # seconds
CROP_END = 6    # seconds

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

# =========================
# PREPROCESS
# =========================
import numpy as np
from scipy.signal import butter, lfilter, resample

def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    return butter(order, [low, high], btype='band')


def preprocess_sample(raw_data):
    sample = raw_data.T
    sample = sample - np.mean(sample, axis=0, keepdims=True)

    # FILTER: Uncomment this to apply filter
    # b, a = butter_bandpass(0.01, 38, 250)
    # sample = lfilter(b, a, sample, axis=-1)

    sample = sample[:, 2 * 250:6 * 250]
    sample = resample(sample, 800, axis=-1)

    return sample.reshape(22, 4, 200)

# =========================
# 2. EXTRACT TRIALS
# =========================
def process_file(file_path):
    data = scipy.io.loadmat(file_path)
    samples = []

    num_runs = len(data['data'][0])

    for j in range(3, num_runs):

        raw = data['data'][0, j][0, 0][0][:, :22]
        events = data['data'][0, j][0, 0][1][:, 0]
        labels = data['data'][0, j][0, 0][2][:, 0]

        events = events.tolist()
        events.append(raw.shape[0])

        for i, ((start, end), label) in enumerate(zip(zip(events[:-1], events[1:]), labels)):

            trial = raw[start:end]

            samples.append({
                "sample": preprocess_sample(trial),
                "label": int(label - 1),
            })

    return samples

def split_file_samples(samples, val_ratio=0.1):
    random.shuffle(samples)

    split_idx = int((1 - val_ratio) * len(samples))

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    return train_samples, val_samples

# =========================
# 4. LOSO NEW PIPELINE
# =========================

def create_loso_splits(data_dir):

    subjects = [f"A0{i}" for i in range(1, 10)]
    splits = []

    for i in range(1, 10):

        test_sub = f"A0{i}"

        train_samples = []
        val_samples = []

        print(f"\nProcessing LOSO subject {test_sub}")

        for j in range(1, 10):

            sub = f"A0{j}"

            T_path = os.path.join(data_dir, f"{sub}T.mat")
            E_path = os.path.join(data_dir, f"{sub}E.mat")

            # ===== TEST SUBJECT =====
            if j == i:
                test_samples = []
                test_samples.extend(process_file(T_path))
                test_samples.extend(process_file(E_path))
                continue

            # ===== TRAIN SUBJECT =====
            # split riêng từng file
            T_samples = process_file(T_path)
            E_samples = process_file(E_path)

            T_train, T_val = split_file_samples(T_samples)
            E_train, E_val = split_file_samples(E_samples)

            train_samples.extend(T_train)
            train_samples.extend(E_train)

            val_samples.extend(T_val)
            val_samples.extend(E_val)

        splits.append({
            "test_subject": test_sub,
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        })

    return splits

def build_lmdb(split_data, save_path, map_size=int(2e9)):
    os.makedirs(save_path, exist_ok=True)

    db = lmdb.open(save_path, map_size=map_size)
    txn = db.begin(write=True)

    keys = {"train": [], "val": [], "test": []}

    for mode in ["train", "val", "test"]:

        print(f"Processing {mode}...")

        for i, item in enumerate(split_data[mode]):
            key = f"{mode}-{i}"

            txn.put(
                key.encode(),
                pickle.dumps({
                    "sample": item["sample"],
                    "label": item["label"]
                })
            )

            keys[mode].append(key)

    txn.put(b"__keys__", pickle.dumps(keys))

    txn.commit()
    db.close()

    print(f"Saved LMDB: {save_path}")

def main():
    DATA_DIR = "BCICIV_2a_gdf"
    # SAVE_ROOT = "datasets/downstream/lmdb_0_38Hz"
    SAVE_ROOT = "datasets/downstream/lmdb"

    splits = create_loso_splits(DATA_DIR)

    for split in splits:

        test_sub = split["test_subject"]

        save_path = os.path.join(SAVE_ROOT, f"LOSO_{test_sub}")

        build_lmdb(split, save_path)


if __name__ == "__main__":
    main()