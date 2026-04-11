import os
import numpy as np
import random
import pickle
import lmdb
from scipy.signal import resample
import mne
import scipy.io as sio

# =========================
# CONFIG
# =========================
SEED = 42
FS = 250
RESAMPLE_POINTS = 800

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

# =========================
# PREPROCESS
# =========================
def preprocess_sample_2b(sample):
    """
    sample: (3, time)
    return: (3, 4, 200)
    """

    # remove DC
    sample = sample - np.mean(sample, axis=1, keepdims=True)

    # resample (1000 → 800)
    sample = resample(sample, RESAMPLE_POINTS, axis=-1)

    # reshape → patches
    sample = sample.reshape(3, 4, 200)

    return sample


# =========================
# LOAD DATA (MNE)
# =========================
class LoadBCIC_2b:

    def __init__(self, path, subject, tmin=2, tmax=6, bandpass=[0, 38]):
        self.path = path
        self.subject = subject
        self.tmin = tmin
        self.tmax = tmax
        self.bandpass = bandpass

        self.train_name = ['1', '2', '3']
        self.test_name = ['4', '5']

        self.train_stim_code = ['769', '770']
        self.test_stim_code = ['783']

        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']

    def get_epoch(self, data_path, isTrain=True):
        raw = mne.io.read_raw_gdf(data_path, preload=True)

        events, event_id = mne.events_from_annotations(raw)

        if isTrain:
            stims = [v for k, v in event_id.items() if k in self.train_stim_code]
        else:
            stims = [v for k, v in event_id.items() if k in self.test_stim_code]

        epochs = mne.Epochs(
            raw,
            events,
            stims,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            preload=True,
            proj=False,
            reject_by_annotation=False
        )

        # filter
        if self.bandpass is not None:
            epochs.filter(self.bandpass[0], self.bandpass[1], method='iir')

        # drop EOG → còn 3 channel
        epochs = epochs.drop_channels(self.channels_to_remove)

        data = epochs.get_data() * 1e6  # µV

        return data  # (N, 3, ~1000)

    def get_label(self, label_path):
        label_info = sio.loadmat(label_path)
        return label_info['classlabel'].reshape(-1) - 1

    def get_train_data(self):
        data, label = [], []

        for se in self.train_name:
            gdf = os.path.join(self.path, f"B0{self.subject}0{se}T.gdf")
            mat = os.path.join(self.path, f"B0{self.subject}0{se}T.mat")

            x = self.get_epoch(gdf, True)
            y = self.get_label(mat)

            data.append(x)
            label.append(y)

        return np.concatenate(data), np.concatenate(label)

    def get_test_data(self):
        data, label = [], []

        for se in self.test_name:
            gdf = os.path.join(self.path, f"B0{self.subject}0{se}E.gdf")
            mat = os.path.join(self.path, f"B0{self.subject}0{se}E.mat")

            x = self.get_epoch(gdf, False)
            y = self.get_label(mat)

            data.append(x)
            label.append(y)

        return np.concatenate(data), np.concatenate(label)


# =========================
# PROCESS → LMDB FORMAT
# =========================
def process_subject(loader):

    train_x, train_y = loader.get_train_data()
    test_x, test_y = loader.get_test_data()

    train_samples = []
    test_samples = []

    # TRAIN
    for x, y in zip(train_x, train_y):
        train_samples.append({
            "sample": preprocess_sample_2b(x),
            "label": int(y)
        })

    # TEST
    for x, y in zip(test_x, test_y):
        test_samples.append({
            "sample": preprocess_sample_2b(x),
            "label": int(y)
        })

    return train_samples, test_samples


# =========================
# LOSO SPLIT
# =========================
def create_loso_splits_2b(data_dir):

    splits = []

    for i in range(1, 10):

        test_sub = f"B0{i}"

        train_samples = []
        val_samples = []
        test_samples = []

        print(f"\nProcessing LOSO subject {test_sub}")

        for j in range(1, 10):

            loader = LoadBCIC_2b(
                path=data_dir,
                subject=j,
                tmin=2,
                tmax=6,
                # bandpass=[0, 38] # đặt filter là 0-38Hz
                bandpass=None # bỏ filter
            )

            sub_train, sub_test = process_subject(loader)

            if j == i:
                test_samples.extend(sub_test)
            else:
                split_idx = int(0.9 * len(sub_train))

                train_samples.extend(sub_train[:split_idx])
                val_samples.extend(sub_train[split_idx:])

        splits.append({
            "test_subject": test_sub,
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        })

    return splits


# =========================
# LMDB
# =========================
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


# =========================
# MAIN
# =========================
def main():
    set_seed()

    DATA_DIR = "/home/infres/ttran-25/project/BCICIV_2b_gdf"
    # SAVE_ROOT = "datasets/downstream/lmdb_2b_0_38Hz"
    SAVE_ROOT = "datasets/downstream/lmdb_2b"


    splits = create_loso_splits_2b(DATA_DIR)

    for split in splits:
        test_sub = split["test_subject"]
        save_path = os.path.join(SAVE_ROOT, f"LOSO_{test_sub}")

        build_lmdb(split, save_path)


if __name__ == "__main__":
    main()