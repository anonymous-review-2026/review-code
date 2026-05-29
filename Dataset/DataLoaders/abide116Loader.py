import torch
import numpy as np

datadir = "./Dataset/Data"


def abide116Loader(atlas, targetTask):
    dataset = torch.load(datadir + "/dataset_abide116_{}.save".format(atlas), weights_only=False)

    x = []
    y = []
    subjectIds = []
    oasCorrs = []

    for data in dataset:
        if targetTask == "disease":
            raw = int(data["pheno"]["disease"])
            label = raw - 1 if raw in (1, 2) else raw
        else:
            raise ValueError(f"abide116Loader only supports targetTask='disease', got {targetTask!r}")

        roi_timeseries_TN = data["roiTimeseries"]
        roi_signal = np.asarray(roi_timeseries_TN).T
        oas = data.get("oasCorr", None)
        if oas is None:
            raise ValueError("ABIDE116 dataset is expected to contain precomputed 'oasCorr' for every sample.")

        x.append(roi_signal)
        y.append(label)
        subjectIds.append(int(data["pheno"].get("subjectId", len(subjectIds))))
        oasCorrs.append(np.asarray(oas, dtype=np.float32))

    return x, y, subjectIds, oasCorrs
