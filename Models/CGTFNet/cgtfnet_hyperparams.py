from utils import Option


def get_hyper_cgtfnet(dataset_name=None):
    hyper_dict = {
        "weightDecay": 0,
        "lr": 2e-4,
        "minLr": 2e-5,
        "maxLr": 4e-4,
        "lrScheduler": "cosine",
        "nOfLayers": 3,
        "dim": 116 if dataset_name in ("adni116", "abide116") else 400,
        "numHeads": 36,
        "headDim": 20,
        "windowSize": 20,
        "shiftSize": 8,
        "fringeSize": 0,
        "fringeCoeff": 24,
        "mlpRatio": 1.0,
        "attentionBias": True,
        "drop": 0.1,
        "attnDrop": 0.1,
        "lambdaCons": 0,
        "labelSmoothing": 0.0,
        "pooling": "cls",
        "useSpdFusion": False,
        "globalSpdEps": 1e-3,
        "crtDropout": 0.3,
        "crtWeightDecay": None,
        "crtWindowAttn": True,
        "crtDetachSpdQ": True,
        "crtAttnNum": 1,
        "crtAttnDim": 32,
        "debugModel": False,
        "debugEvery": 20,
        "debugStepLogs": False,
        "seed": None,
    }
    return Option(hyper_dict)
