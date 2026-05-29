from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

from .DataLoaders.adni116Loader import adni116Loader
from .DataLoaders.abide116Loader import abide116Loader

loaderMapper = {
    "adni116": adni116Loader,
    "abide116": abide116Loader,
}


def _load_or_create_precomputed_folds(labels, splitFile, foldCount, splitSeed):
    splitPath = Path(splitFile)
    labels = np.asarray(labels, dtype=np.int64)

    if splitPath.exists():
        saved = np.load(splitPath, allow_pickle=False)
        saved_labels = np.asarray(saved["labels"], dtype=np.int64)
        if not np.array_equal(saved_labels, labels):
            raise ValueError(
                f"Saved split file {splitPath} does not match current labels. "
                f"Expected shape={labels.shape}, found shape={saved_labels.shape}."
            )
        return [(saved[f"train_idx_{fold_idx}"], saved[f"test_idx_{fold_idx}"]) for fold_idx in range(foldCount)]

    skf = StratifiedKFold(n_splits=foldCount, shuffle=True, random_state=splitSeed)
    dummy_x = np.zeros(labels.shape[0], dtype=np.float32)
    save_items = {"labels": labels}
    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dummy_x, labels)):
        train_idx = train_idx.astype(np.int64, copy=False)
        test_idx = test_idx.astype(np.int64, copy=False)
        save_items[f"train_idx_{fold_idx}"] = train_idx
        save_items[f"test_idx_{fold_idx}"] = test_idx
        folds.append((train_idx, test_idx))

    splitPath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(splitPath, **save_items)
    print(f"Created split file: {splitPath}")
    return folds


def getDataset(options):
    return SupervisedDataset(options)


class SupervisedDataset(Dataset):
    def __init__(self, datasetDetails):
        self.batchSize = datasetDetails.batchSize
        self.dynamicLength = datasetDetails.dynamicLength
        self.foldCount = datasetDetails.foldCount
        self.seed = datasetDetails.datasetSeed
        self.splitFile = getattr(datasetDetails, "splitFile", None)
        self.splitSeed = int(getattr(datasetDetails, "splitSeed", 42))

        loader = loaderMapper[datasetDetails.datasetName]
        self.precomputedFolds = None
        self.kFold = StratifiedKFold(datasetDetails.foldCount, shuffle=False, random_state=None) if datasetDetails.foldCount is not None else None
        self.k = None

        loaded = loader(datasetDetails.atlas, datasetDetails.targetTask)
        self.data, self.labels, self.subjectIds, self.oasCorrs = loaded

        self.fullDynamicLength = int(self.data[0].shape[-1]) if len(self.data) > 0 else None
        if self.dynamicLength is None and self.fullDynamicLength is not None:
            if any(int(subject.shape[-1]) != self.fullDynamicLength for subject in self.data):
                raise ValueError(f"{datasetDetails.datasetName} has inconsistent sequence lengths.")
            self.dynamicLength = self.fullDynamicLength

        if self.splitFile is not None and self.foldCount is not None:
            self.precomputedFolds = _load_or_create_precomputed_folds(self.labels, self.splitFile, self.foldCount, self.splitSeed)
            self.kFold = None
        else:
            random.Random(self.seed).shuffle(self.data)
            random.Random(self.seed).shuffle(self.labels)
            random.Random(self.seed).shuffle(self.subjectIds)
            random.Random(self.seed).shuffle(self.oasCorrs)

        self.targetData = None
        self.targetLabels = None
        self.targetSubjIds = None
        self.targetOasCorrs = None
        self.randomRanges = None

    def __len__(self):
        return len(self.data) if self.targetData is None else len(self.targetData)

    def get_nOfTrains_perFold(self):
        if self.precomputedFolds is not None:
            return int(len(self.precomputedFolds[0][0]))
        if self.foldCount is not None:
            return int(np.ceil(len(self.data) * (self.foldCount - 1) / self.foldCount))
        return len(self.data)

    def setFold(self, fold, train=True):
        self.k = fold
        self.train = train

        if self.foldCount is None:
            trainIdx = list(range(len(self.data)))
            testIdx = []
        elif self.precomputedFolds is not None:
            trainIdx, testIdx = self.precomputedFolds[fold]
        else:
            trainIdx, testIdx = list(self.kFold.split(self.data, self.labels))[fold]

        if self.precomputedFolds is None:
            trainIdx = np.asarray(trainIdx).copy()
            random.Random(self.seed).shuffle(trainIdx)

        self.targetData = [self.data[idx] for idx in trainIdx] if train else [self.data[idx] for idx in testIdx]
        self.targetLabels = [self.labels[idx] for idx in trainIdx] if train else [self.labels[idx] for idx in testIdx]
        self.targetSubjIds = [self.subjectIds[idx] for idx in trainIdx] if train else [self.subjectIds[idx] for idx in testIdx]
        self.targetOasCorrs = [self.oasCorrs[idx] for idx in trainIdx] if train else [self.oasCorrs[idx] for idx in testIdx]

        if train and self.dynamicLength is not None:
            np.random.seed(self.seed + 1)
            self.randomRanges = []
            for idx in trainIdx:
                subjectLength = int(self.data[idx].shape[-1])
                maxInit = subjectLength - self.dynamicLength
                if maxInit == 0:
                    starts = [0 for _ in range(9999)]
                else:
                    starts = [np.random.randint(0, maxInit + 1) for _ in range(9999)]
                self.randomRanges.append(starts)

    def getFold(self, fold, train=True):
        self.setFold(fold, train)
        if train:
            return DataLoader(self, batch_size=self.batchSize, shuffle=(self.precomputedFolds is not None))
        return DataLoader(self, batch_size=1, shuffle=False)

    def __getitem__(self, idx):
        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]

        timeseries = np.asarray(subject, dtype=np.float32)
        roi_mean = np.mean(timeseries, axis=1, keepdims=True)
        roi_std = np.std(timeseries, axis=1, keepdims=True)
        safe_std_mask = roi_std > 1e-6
        centered = timeseries - roi_mean
        timeseries = np.divide(centered, roi_std, out=np.zeros_like(centered, dtype=np.float32), where=safe_std_mask)
        timeseries = np.nan_to_num(timeseries, 0.0)

        if self.train and self.dynamicLength is not None:
            samplingInit = self.randomRanges[idx].pop()
            timeseries = timeseries[:, samplingInit : samplingInit + self.dynamicLength]

        batch = {"timeseries": timeseries.astype(np.float32), "label": label, "subjId": subjId}
        batch["oasCorr"] = np.asarray(self.targetOasCorrs[idx], dtype=np.float32)
        return batch
