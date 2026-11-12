import os
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from datasets import load_dataset as load_hf_dataset
from pathlib import Path

from PIL import Image



def _cache_dir():
    return os.environ.get("LEVERAGE_CACHE_DIR", os.path.abspath("./cache"))


def _hf_cache():
    return os.path.join(_cache_dir(), "huggingface")



class Cifar10Dataset(Dataset):
    def __init__(self, root_dir="./data/Cifar10", train=True, download=True):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        self.dataset = datasets.CIFAR10(
            root=root_dir, train=train, download=download, transform=self.transform
        )

    @property
    def labels(self):
        return np.array(self.dataset.targets)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)


class Cifar100Dataset(Dataset):
    def __init__(self, root='./data/Cifar100', train=True, download=True):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                 std=[0.2673, 0.2564, 0.2762])
        ])
        self.dataset = datasets.CIFAR100(
            root=root, train=train, download=download, transform=self.transform
        )

    @property
    def labels(self):
        return np.array(self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx


class SST2Dataset(Dataset):

    def __init__(self, split="train", model_name="google/bert_uncased_L-2_H-128_A-2", cache_dir="./cache"):
        super().__init__()
        dataset_cache = Path(cache_dir) /  "huggingface/datasets"
        

        # Note: 'load_hf_dataset' is likely a wrapper for datasets.load_dataset
        self.hf_dataset = load_hf_dataset(
            "glue", "sst2",
            cache_dir=dataset_cache,
            split=split,
        )


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=Path(cache_dir) / "huggingface",
        )
    @property
    def labels(self):
        return np.array(self.hf_dataset['label'])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[int(idx)]
        encoding = self.tokenizer(
            item['sentence'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, item['label'], idx


class CaliforniaHousingDataset(Dataset):
    """
    California Housing regression dataset (sklearn).
    20,640 samples, 8 continuous features, continuous target (median house value).
    Task: regression.

    Splits are deterministic via a fixed random_state.
    Downloads, processes, and saves the data locally on the first run, 
    then loads from disk on subsequent runs to save time.
    """
    def __init__(self, data_dir="./data/CaliforniaHousing", train=True, test_size=0.2, random_state=42, scale=True):
        super().__init__()
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        file_name = f"california_housing_ts{test_size}_rs{random_state}_scale{scale}.pt"
        self.file_path = os.path.join(self.data_dir, file_name)

        # Check if the processed dataset already exists locally
        if not os.path.exists(self.file_path):
            self._download_and_process(test_size, random_state, scale)

        # Load the saved tensors from disk
        data = torch.load(self.file_path)
        
        if train:
            self.X = data['X_train']
            self.y = data['y_train']
        else:
            self.X = data['X_test']
            self.y = data['y_test']

    def _download_and_process(self, test_size, random_state, scale):
        print(f"Downloading and processing dataset to {self.file_path}...")
        raw = fetch_california_housing()
        X, y = raw.data.astype(np.float32), raw.target.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # Save all splits to a single dictionary
        torch.save({
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.float32),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.float32)
        }, self.file_path)
        print("Dataset saved successfully.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # target reshaped to [1] to safely match MLP output
        return self.X[idx], self.y[idx].view(1), idx



_UCI_CONFIGS = {
    # (url_or_sklearn_loader, target_column_or_None, task)
    "diabetes": {
        "source": "sklearn",          # uses sklearn.datasets.load_diabetes
        "task": "regression",
        "n_classes": None,
    },
    "energy": {
        # UCI Energy Efficiency dataset — predicts heating load (Y1)
        "source": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "target_col": "Y1",
        "drop_cols": ["Y2"],
        "task": "regression",
        "n_classes": None,
    },
}

class UCIDataset(Dataset):
    """
    Generic wrapper for UCI regression / classification datasets.

    Supported names:  'diabetes' | 'energy'

    Usage:
        train_ds = UCIDataset("energy", train=True)
        test_ds  = UCIDataset("energy", train=False)
    """
    def __init__(self, name: str, train=True, test_size=0.2,
                 random_state=42, root_dir="./data/UCI", scale=True):
        super().__init__()
        assert name in _UCI_CONFIGS, f"Unknown UCI dataset '{name}'. Choose from {list(_UCI_CONFIGS)}"
        cfg = _UCI_CONFIGS[name]
        self.task     = cfg["task"]
        self.n_classes = cfg.get("n_classes")

        X, y = self._load(name, cfg, root_dir)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=(y if self.task == "classification" else None)
        )

        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test  = self.scaler.transform(X_test)

        self.X = torch.tensor(X_train if train else X_test, dtype=torch.float32)
        dtype_y = torch.long if self.task == "classification" else torch.float32
        self.y = torch.tensor(y_train if train else y_test, dtype=dtype_y)

    # ------------------------------------------------------------------
    def _load(self, name, cfg, root_dir):
        if cfg["source"] == "sklearn":
            raw = load_diabetes()
            return raw.data.astype(np.float32), raw.target.astype(np.float32)

        os.makedirs(root_dir, exist_ok=True)

        if name == "energy":
            fpath = os.path.join(root_dir, "energy_data.csv")
            
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing {fpath}. Please run the CSV conversion script first!")
            
            # Now we use read_csv! No openpyxl required.
            df = pd.read_csv(fpath).dropna() 
            
            drop_cols = cfg.get("drop_cols", [])
            y = df[cfg["target_col"]].values.astype(np.float32)
            X = df.drop(columns=[cfg["target_col"]] + drop_cols).values.astype(np.float32)
            return X, y

        raise ValueError(f"Unhandled UCI config for '{name}'")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx



class Purchase100Dataset(Dataset):
    """
    Purchase-100 dataset from Shokri et al. (2017).
    ~197,324 samples, 600 binary features, 100 classes.
    Task: classification.

    The preprocessed .npz is fetched from the canonical mirror used in
    the MIA literature. If the file already exists in root_dir it is reused.

    Source: https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz
    Preprocessed npz mirror (Carlini et al. release):
        https://storage.googleapis.com/mlperf-pubs/purchase100.npz   (primary)
    Fallback: process raw tgz yourself (see _process_raw below).
    """
    _NPZ_URL = "https://storage.googleapis.com/mlperf-pubs/purchase100.npz"
    _TGZ_URL = "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz"

    def __init__(self, train=True, test_size=0.2, random_state=42,
                 root_dir="./data/Purchase100"):
        super().__init__()
        os.makedirs(root_dir, exist_ok=True)
        npz_path = os.path.join(root_dir, "purchase100.npz")

        if not os.path.exists(npz_path):
            self._download_or_process(root_dir, npz_path)

        data = np.load(npz_path)
        X = data["X"].astype(np.float32)   # (N, 600)  binary features
        y = data["y"].astype(np.int64)     # (N,)      0-indexed class labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.X = torch.tensor(X_train if train else X_test, dtype=torch.float32)
        self.y = torch.tensor(y_train if train else y_test, dtype=torch.long)

    # ------------------------------------------------------------------
    def _download_or_process(self, root_dir, npz_path):
        """Try the pre-built npz first; fall back to processing the raw tgz."""
        try:
            print("Downloading Purchase-100 .npz …")
            urllib.request.urlretrieve(self._NPZ_URL, npz_path)
        except Exception:
            print("Primary mirror failed. Downloading raw tgz and processing …")
            self._process_raw(root_dir, npz_path)

    def _process_raw(self, root_dir, npz_path):
        import tarfile
        tgz_path = os.path.join(root_dir, "dataset_purchase.tgz")
        urllib.request.urlretrieve(self._TGZ_URL, tgz_path)
        with tarfile.open(tgz_path) as t:
            t.extractall(root_dir)
        # The raw file is a CSV: label,f1,f2,...,f600
        raw_csv = os.path.join(root_dir, "dataset_purchase")
        df = pd.read_csv(raw_csv, header=None)
        y = df.iloc[:, 0].values.astype(np.int64) - 1   # 1-indexed → 0-indexed
        X = df.iloc[:, 1:].values.astype(np.float32)
        np.savez_compressed(npz_path, X=X, y=y)
        print(f"Saved processed Purchase-100 to {npz_path}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx



class Texas100Dataset(Dataset):
    """
    Texas-100 hospital discharge dataset from Shokri et al. (2017).
    ~67,330 samples, 6,169 binary features, 100 classes (procedure codes).
    Task: classification.

    Preprocessed .npz fetched from the canonical mirror.
    Source: Texas Department of State Health Services.
    Preprocessed version: https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz
    """
    _TGZ_URL = "https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz"

    def __init__(self, train=True, test_size=0.2, random_state=42,
                 root_dir="./data/Texas100"):
        super().__init__()
        os.makedirs(root_dir, exist_ok=True)
        npz_path = os.path.join(root_dir, "texas100.npz")

        if not os.path.exists(npz_path):
            self._download_and_process(root_dir, npz_path)

        data = np.load(npz_path)
        X = data["X"].astype(np.float32)   # (N, 6169) binary features
        y = data["y"].astype(np.int64)     # (N,)      0-indexed class labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.X = torch.tensor(X_train if train else X_test, dtype=torch.float32)
        self.y = torch.tensor(y_train if train else y_test, dtype=torch.long)

    # ------------------------------------------------------------------
    def _download_and_process(self, root_dir, npz_path):
        import tarfile
        print("Downloading Texas-100 tgz …")
        tgz_path = os.path.join(root_dir, "dataset_texas.tgz")
        urllib.request.urlretrieve(self._TGZ_URL, tgz_path)
        with tarfile.open(tgz_path) as t:
            t.extractall(root_dir)

        # Raw file layout: first col = label (1-indexed), remaining = binary features
        raw_path = os.path.join(root_dir, "dataset_texas", "feats")
        label_path = os.path.join(root_dir, "dataset_texas", "labels")

        if os.path.exists(raw_path):                 # separate feats / labels files
            X = np.load(raw_path).astype(np.float32)
            y = np.load(label_path).astype(np.int64) - 1
        else:                                        # single CSV fallback
            raw_csv = os.path.join(root_dir, "dataset_texas")
            df = pd.read_csv(raw_csv, header=None)
            y = df.iloc[:, 0].values.astype(np.int64) - 1
            X = df.iloc[:, 1:].values.astype(np.float32)

        np.savez_compressed(npz_path, X=X, y=y)
        print(f"Saved processed Texas-100 to {npz_path}")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx




class MedicalDataset(Dataset):
    """
    Medical prediction datasets.  Three sub-tasks are supported:

        "heart_failure"  — Heart Failure Clinical Records (Kaggle / UCI)
                           299 samples, 12 clinical features, binary (death event).
                           Task: classification.

        "breast_cancer"  — Wisconsin Diagnostic Breast Cancer (sklearn)
                           569 samples, 30 features, binary (malignant / benign).
                           Task: classification.

        "diabetes_readmission"
                         — UCI Diabetes 130-US hospitals dataset.
                           ~100k samples, readmitted <30 days (binary).
                           Task: classification.

    Usage:
        ds = MedicalDataset("heart_failure", train=True)
    """
    _HEART_FAILURE_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00519/heart_failure_clinical_records_dataset.csv"
    )

    def __init__(self, name: str = "breast_cancer", train=True,
                 test_size=0.2, random_state=42,
                 root_dir="./data/Medical", scale=True):
        super().__init__()
        os.makedirs(root_dir, exist_ok=True)
        assert name in ("heart_failure", "breast_cancer", "diabetes_readmission"), \
            f"Unknown medical dataset '{name}'"

        X, y = self._load(name, root_dir)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test  = self.scaler.transform(X_test)

        self.X = torch.tensor(X_train if train else X_test, dtype=torch.float32)
        self.y = torch.tensor(y_train if train else y_test, dtype=torch.long)

    def _load(self, name, root_dir):
        if name == "breast_cancer":
            from sklearn.datasets import load_breast_cancer
            raw = load_breast_cancer()
            return raw.data.astype(np.float32), raw.target.astype(np.int64)

        if name == "heart_failure":
            fpath = os.path.join(root_dir, "heart_failure.csv")
            if not os.path.exists(fpath):
                print("Downloading Heart Failure dataset …")
                urllib.request.urlretrieve(self._HEART_FAILURE_URL, fpath)
            df = pd.read_csv(fpath)
            y = df["DEATH_EVENT"].values.astype(np.int64)
            X = df.drop(columns=["DEATH_EVENT"]).values.astype(np.float32)
            return X, y

        if name == "diabetes_readmission":
            return self._load_diabetes_readmission(root_dir)

    def _load_diabetes_readmission(self, root_dir):
        """
        UCI Diabetes 130-US Hospitals dataset.
        Binarised target: readmitted within <30 days (1) vs otherwise (0).
        Downloaded via UCI ML Repo.
        """
        fpath = os.path.join(root_dir, "diabetic_data.csv")
        if not os.path.exists(fpath):
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "00296/dataset_diabetes.zip"
            )
            zip_path = os.path.join(root_dir, "dataset_diabetes.zip")
            print("Downloading Diabetes 130-US Hospitals dataset …")
            urllib.request.urlretrieve(url, zip_path)
            import zipfile
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(root_dir)

        df = pd.read_csv(fpath, na_values="?").dropna()

        # Binary target: readmitted < 30 days
        df["target"] = (df["readmitted"] == "<30").astype(int)
        df = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"])

        # Encode categoricals
        for col in df.select_dtypes(include="object").columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        y = df["target"].values.astype(np.int64)
        X = df.drop(columns=["target"]).values.astype(np.float32)
        return X, y

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx



class BloodMnistDataset(Dataset):
    """
    BloodMNIST from the MedMNIST benchmark (Standalone Version).
    ~17k microscopy images of blood cells (28x28, resized to 224 for DINOv2).
    8 classes. Task: classification.
    """
    def __init__(self, split="train", download=False, root_dir="./data/BloodMNIST"):
        super().__init__()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        file_name = f'bloodmnist_224.npz' 
        data_path = os.path.join(root_dir, file_name)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data not found at {data_path}. "
            )

        dataset = np.load(data_path)
        
        if split == 'train':
            self.images = dataset['train_images']
            self._labels = dataset['train_labels']
        elif split == 'val':
            self.images = dataset['val_images']
            self._labels = dataset['val_labels']
        elif split == 'test':
            self.images = dataset['test_images']
            self._labels = dataset['test_labels']
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

    @property
    def labels(self):
        return self._labels.flatten().astype(int)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_array = self.images[idx]
        label = self.labels[idx]
        
        img = Image.fromarray(img_array)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx

def get_dataset(dataset_name, **kwargs):
    registry = {
        "cifar10":      lambda: Cifar10Dataset(**kwargs),
        "cifar100":     lambda: Cifar100Dataset(**kwargs),
        "sst2":         lambda: SST2Dataset(
                            model_name=kwargs.get("bert_model_name", "google/bert_uncased_L-2_H-128_A-2"),
                            split=kwargs.get("split", "train"),
                            cache_dir=kwargs.get("cache_dir", "./cache")),
        "california_housing": lambda: CaliforniaHousingDataset(**kwargs),
        "uci":          lambda: UCIDataset(
                            name=kwargs.get("sub_dataset", "energy"), **kwargs),
        "purchase100":  lambda: Purchase100Dataset(**kwargs),
        "texas100":     lambda: Texas100Dataset(**kwargs),
        "medical":      lambda: MedicalDataset(
                            name=kwargs.pop("name", "breast_cancer"), **kwargs),
        "blood_mnist": lambda: BloodMnistDataset(**kwargs
            
        )
    }
    if dataset_name not in registry:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[dataset_name]()