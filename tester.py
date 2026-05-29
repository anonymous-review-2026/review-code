import argparse
import torch
import warnings

from utils import Option, metricSummer, calculateMetrics, dumpTestResults
from Dataset.datasetDetails import datasetDetailsDict
from Models.CGTFNet.run_cgtfnet import run_cgtfnet
from Models.CGTFNet.cgtfnet_hyperparams import get_hyper_cgtfnet

warnings.filterwarnings("ignore", message="Precision is ill-defined*", category=UserWarning)

parser = argparse.ArgumentParser()


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def int_or_none(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    v = str(v).strip().lower()
    if v in ("none", "null", "full", "all"):
        return None
    return int(v)


parser.add_argument("-d", "--dataset", type=str, default="adni116")
parser.add_argument("-m", "--model", type=str, default="cgtfnet")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default="noname")
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--folds", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--dynamic_length", type=int_or_none, default="__KEEP_DEFAULT__")
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--shift_size", type=int, default=None)
parser.add_argument("--fringe_size", type=int, default=None)
parser.add_argument("--fringe_coeff", type=int, default=None)
parser.add_argument("--n_layers", type=int, default=None)
parser.add_argument("--num_heads", type=int, default=None)
parser.add_argument("--head_dim", type=int, default=None)
parser.add_argument("--use_spd_fusion", type=str2bool, default=None)
parser.add_argument("--crt_dropout", type=float, default=None)
parser.add_argument("--crt_window_attn", type=str2bool, default=None)
parser.add_argument("--crt_attn_dim", type=int, default=None)
parser.add_argument("--crt_attnum", type=int, default=None)
parser.add_argument("--crt_detach_spd_q", type=str2bool, default=None)
parser.add_argument("--debug_model", type=str2bool, default=None)
parser.add_argument("--debug_every", type=int, default=None)
parser.add_argument("--debug_step_logs", type=str2bool, default=None)
parser.add_argument("--lr_scheduler", type=str, default=None, choices=("none", "cosine"))
parser.add_argument("--label_smoothing", type=float, default=None)
parser.add_argument("--crt_weight_decay", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)

argv = parser.parse_args()

if argv.model != "cgtfnet":
    raise ValueError("CGTFNet-anon 仅保留 CGTFNet 相关链路，请使用 -m cgtfnet")

print("\nTest model is {}".format(argv.model))

datasetName = argv.dataset
datasetDetails = dict(datasetDetailsDict[datasetName])
if argv.epochs is not None:
    datasetDetails["nOfEpochs"] = argv.epochs
if argv.folds is not None:
    datasetDetails["foldCount"] = argv.folds
if argv.batch_size is not None:
    datasetDetails["batchSize"] = argv.batch_size
if argv.dynamic_length != "__KEEP_DEFAULT__":
    datasetDetails["dynamicLength"] = argv.dynamic_length

hyperParams = get_hyper_cgtfnet(datasetName)

if argv.window_size is not None:
    hyperParams.windowSize = int(argv.window_size)
if argv.shift_size is not None:
    hyperParams.shiftSize = int(argv.shift_size)
if argv.fringe_size is not None:
    hyperParams.fringeSize = int(argv.fringe_size)
if argv.fringe_coeff is not None:
    hyperParams.fringeCoeff = int(argv.fringe_coeff)
if argv.n_layers is not None:
    hyperParams.nOfLayers = int(argv.n_layers)
if argv.num_heads is not None:
    hyperParams.numHeads = int(argv.num_heads)
if argv.head_dim is not None:
    hyperParams.headDim = int(argv.head_dim)
if argv.use_spd_fusion is not None:
    hyperParams.useSpdFusion = bool(argv.use_spd_fusion)
if argv.crt_dropout is not None:
    hyperParams.crtDropout = float(argv.crt_dropout)
if argv.crt_window_attn is not None:
    hyperParams.crtWindowAttn = bool(argv.crt_window_attn)
if argv.crt_attn_dim is not None:
    hyperParams.crtAttnDim = int(argv.crt_attn_dim)
if argv.crt_attnum is not None:
    hyperParams.crtAttnNum = max(1, int(argv.crt_attnum))
if argv.crt_detach_spd_q is not None:
    hyperParams.crtDetachSpdQ = bool(argv.crt_detach_spd_q)
if argv.debug_model is not None:
    hyperParams.debugModel = bool(argv.debug_model)
if argv.debug_every is not None:
    hyperParams.debugEvery = int(argv.debug_every)
if argv.debug_step_logs is not None:
    hyperParams.debugStepLogs = bool(argv.debug_step_logs)
if argv.lr_scheduler is not None:
    hyperParams.lrScheduler = argv.lr_scheduler
if argv.label_smoothing is not None:
    hyperParams.labelSmoothing = float(argv.label_smoothing)
if bool(getattr(hyperParams, "useSpdFusion", False)):
    hyperParams.lr = 1e-3
    hyperParams.weightDecay = 1e-4
    if argv.epochs is None:
        datasetDetails["nOfEpochs"] = 50
if argv.crt_weight_decay is not None:
    hyperParams.crtWeightDecay = float(argv.crt_weight_decay)
if argv.seed is not None:
    hyperParams.seed = int(argv.seed)

print(
    "CGTFNet switches : useSpdFusion={}, crtWindowAttn={}, crtAttnNum={}, crtAttnDim={}, crtDropout={}, debugModel={}, debugEvery={}, debugStepLogs={}".format(
        getattr(hyperParams, "useSpdFusion", None),
        getattr(hyperParams, "crtWindowAttn", None),
        getattr(hyperParams, "crtAttnNum", None),
        getattr(hyperParams, "crtAttnDim", None),
        getattr(hyperParams, "crtDropout", None),
        getattr(hyperParams, "debugModel", None),
        getattr(hyperParams, "debugEvery", None),
        getattr(hyperParams, "debugStepLogs", None),
    )
)
print(
    "Temporal encoder geometry : windowSize={}, shiftSize={}, fringeSize={}, fringeCoeff={}, nOfLayers={}, runSeed={}".format(
        getattr(hyperParams, "windowSize", None),
        getattr(hyperParams, "shiftSize", None),
        getattr(hyperParams, "fringeSize", None),
        getattr(hyperParams, "fringeCoeff", None),
        getattr(hyperParams, "nOfLayers", None),
        getattr(hyperParams, "seed", None),
    )
)
_crt_wd = getattr(hyperParams, "crtWeightDecay", None)
_crt_wd_show = getattr(hyperParams, "weightDecay", None) if _crt_wd is None else _crt_wd
print(
    "CGTFNet optimizer : lr={}, minLr={}, weightDecay={}, crtWeightDecay(eff)={}, lrScheduler={}, labelSmoothing={}".format(
        getattr(hyperParams, "lr", None),
        getattr(hyperParams, "minLr", None),
        getattr(hyperParams, "weightDecay", None),
        _crt_wd_show if bool(getattr(hyperParams, "useSpdFusion", False)) else "n/a",
        getattr(hyperParams, "lrScheduler", "none"),
        getattr(hyperParams, "labelSmoothing", 0.0),
    )
)
print("Dataset details : {}".format(datasetDetails))

if datasetName in ("adni116", "abide116"):
    seeds = [42]
else:
    seeds = [0]

resultss = []
for seed in seeds:
    if getattr(hyperParams, "seed", None) is not None:
        outer_seed = int(hyperParams.seed)
    else:
        outer_seed = seed
    torch.manual_seed(outer_seed)
    print("Running the model with seed : {}".format(outer_seed))

    results = run_cgtfnet(
        hyperParams,
        Option({**datasetDetails, "datasetSeed": outer_seed}),
        device="cuda:{}".format(argv.device),
    )
    resultss.append(results)

metricss = calculateMetrics(resultss)
meanMetrics_seeds, stdMetrics_seeds, meanMetric_all, stdMetric_all = metricSummer(metricss, "test")

dumpTestResults(argv.name, hyperParams, argv.model, datasetName, metricss)

print("\n \ n meanMetrics_all : {}".format(meanMetric_all))
print("stdMetric_all : {}".format(stdMetric_all))
