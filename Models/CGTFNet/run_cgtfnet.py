from tqdm import tqdm
import torch
import numpy as np
import random

from utils import Option, calculateMetric
from Models.CGTFNet.cgtfnet_model import CGTFNetModel
from Dataset.dataset import getDataset


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    if probs.ndim != 2 or probs.shape[1] < 2:
        return float("nan")
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        acc_bin = np.mean(accuracies[mask])
        conf_bin = np.mean(confidences[mask])
        ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def _std_from_mean_sq(mean_val: float, sq_mean_val: float) -> float:
    if mean_val != mean_val or sq_mean_val != sq_mean_val:
        return float("nan")
    v = sq_mean_val - mean_val * mean_val
    return float(np.sqrt(max(0.0, v)))


def run_on_loader(model, dataLoader, train: bool, desc: str):
    preds_list = []
    probs_list = []
    labels_list = []
    losses = []
    loss_infos = []

    for _, data in enumerate(tqdm(dataLoader, ncols=60, desc=desc, disable=True)):
        x = data["timeseries"]
        y = data["label"]
        oas = data.get("oasCorr", None)
        loss, preds, probs, y, loss_info = model.step(x, y, train=train, oasCorr=oas)
        torch.cuda.empty_cache()
        preds_list.append(preds)
        probs_list.append(probs)
        labels_list.append(y)
        losses.append(loss)
        loss_infos.append(loss_info)

    preds = torch.cat(preds_list, dim=0).numpy()
    probs = torch.cat(probs_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    loss_mean = float(torch.tensor(losses).numpy().mean())
    metrics = calculateMetric({"predictions": preds, "probs": probs, "labels": labels})

    loss_stats = {}
    if loss_infos:
        total_samples = float(sum(info.get("sampleCount", 1.0) for info in loss_infos))
        for key in loss_infos[0].keys():
            if key == "sampleCount":
                continue
            if key.endswith("Count"):
                loss_stats[key] = float(sum(info[key] for info in loss_infos))
            else:
                loss_stats[key] = float(sum(info[key] * info.get("sampleCount", 1.0) for info in loss_infos) / max(total_samples, 1.0))
    return preds, probs, labels, loss_mean, metrics, loss_stats


def train_and_eval_5fold_style(model, dataset_train, dataset_test, fold: int, nOfEpochs: int, fold_idx_1based: int):
    dataLoader_train = dataset_train.getFold(fold, train=True)
    dataLoader_test = dataset_test.getFold(fold, train=False)
    train_labels_all = np.asarray(dataset_train.targetLabels, dtype=np.int64)
    test_labels_all = np.asarray(dataset_test.targetLabels, dtype=np.int64)

    def _format_ratio(labels_arr: np.ndarray) -> str:
        pos = int(np.sum(labels_arr == 1))
        neg = int(np.sum(labels_arr == 0))
        total = len(labels_arr)
        ratio = pos / total if total > 0 else float("nan")
        return f"pos={pos}, neg={neg}, pos_ratio={ratio:.4f}"

    print("\n" + "=" * 80)
    print(f"Fold {fold_idx_1based}/{dataset_train.foldCount}")
    print("=" * 80)
    print(f"Train labels     : {_format_ratio(train_labels_all)}")
    print(f"Test labels      : {_format_ratio(test_labels_all)}")

    best_test_acc = 0.0
    best_test_auc = float("-inf")
    best_test_recall = float("nan")
    best_test_spec = float("nan")
    final_test_acc = 0.0
    final_test_auc = float("nan")
    final_test_recall = float("nan")
    final_test_spec = float("nan")

    epoch_test_accs = []
    epoch_test_aucs = []
    epoch_test_recalls = []
    epoch_test_specs = []

    best_test_preds = None
    best_test_probs = None
    best_test_labels = None
    best_test_loss = None
    last_train_preds = None
    last_train_probs = None
    last_train_labels = None
    last_train_loss = None

    for epoch in range(1, nOfEpochs + 1):
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        train_preds, train_probs, train_labels, train_loss_mean, train_metrics, train_loss_stats = run_on_loader(
            model, dataLoader_train, train=True, desc=f"fold:{fold} epoch:{epoch}"
        )
        train_acc = float(train_metrics["accuracy"])
        test_preds, test_probs, test_labels, test_loss_mean, test_metrics, test_loss_stats = run_on_loader(
            model, dataLoader_test, train=False, desc=f"Testing fold:{fold} epoch:{epoch}"
        )
        test_acc = float(test_metrics["accuracy"])
        test_auc_val = test_metrics["roc"]
        test_auc = float(test_auc_val) if not np.isnan(test_auc_val) else float("nan")
        test_recall = float(test_metrics["recall"])
        test_spec_val = test_metrics.get("specificity", float("nan"))
        test_spec = float(test_spec_val) if not np.isnan(test_spec_val) else float("nan")

        train_ece = float("nan")
        test_ece = float("nan")
        if train_probs.ndim == 2 and train_probs.shape[1] == 2:
            train_ece = binary_ece(train_probs, train_labels)
            test_ece = binary_ece(test_probs, test_labels)

        epoch_test_accs.append(test_acc)
        epoch_test_aucs.append(test_auc)
        epoch_test_recalls.append(test_recall)
        epoch_test_specs.append(test_spec)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_auc = test_auc
            best_test_recall = test_recall
            best_test_spec = test_spec
            best_test_preds = test_preds
            best_test_probs = test_probs
            best_test_labels = test_labels
            best_test_loss = test_loss_mean

        final_test_acc = test_acc
        final_test_auc = test_auc
        final_test_recall = test_recall
        final_test_spec = test_spec
        last_train_preds = train_preds
        last_train_probs = train_probs
        last_train_labels = train_labels
        last_train_loss = train_loss_mean

        line = (
            f"Epoch [{epoch:03d}/{nOfEpochs}] "
            f"Train loss={train_loss_mean:.4f} "
            f"Test loss={test_loss_mean:.4f} "
            f"Train acc={train_acc:.4f} "
            f"Test acc={test_acc:.4f} "
            f"Test auc={test_auc:.4f} "
            f"gap_acc={train_acc - test_acc:.4f} "
            f"ECE_tr={train_ece:.4f} ECE_te={test_ece:.4f} "
            f"TrainL(F/C)={train_loss_stats.get('finalLossWeighted', float('nan')):.4f}/{train_loss_stats.get('consistencyLossWeighted', float('nan')):.4f} "
            f"TestL(F/C)={test_loss_stats.get('finalLossWeighted', float('nan')):.4f}/{test_loss_stats.get('consistencyLossWeighted', float('nan')):.4f}"
        )
        if getattr(model, "useSpdFusion", False):
            line += (
                f" TrainFus(m/v/n/r)={train_loss_stats.get('fusionLogitMean', float('nan')):.4f}/{train_loss_stats.get('fusionLogitVar', float('nan')):.4f}/{train_loss_stats.get('fusionLogitNorm', float('nan')):.4f}/{train_loss_stats.get('fusionLogitNormToSpdFeatNorm', float('nan')):.4f} "
                f"TestFus(m/v/n/r)={test_loss_stats.get('fusionLogitMean', float('nan')):.4f}/{test_loss_stats.get('fusionLogitVar', float('nan')):.4f}/{test_loss_stats.get('fusionLogitNorm', float('nan')):.4f}/{test_loss_stats.get('fusionLogitNormToSpdFeatNorm', float('nan')):.4f} "
                f"TrainFus(cal)={train_loss_stats.get('fusionPredConfMean', float('nan')):.4f}/{train_loss_stats.get('fusionPredEntropyMean', float('nan')):.4f}/{train_loss_stats.get('fusionLogitMarginMean', float('nan')):.4f} "
                f"TestFus(cal)={test_loss_stats.get('fusionPredConfMean', float('nan')):.4f}/{test_loss_stats.get('fusionPredEntropyMean', float('nan')):.4f}/{test_loss_stats.get('fusionLogitMarginMean', float('nan')):.4f}"
            )
            snap = model.get_debug_snapshot() if hasattr(model, "get_debug_snapshot") else {}
            if snap.get("correctionWeightNormZTemp") is not None:
                line += (
                    f" |W|_z={snap['correctionWeightNormZTemp']:.4f} "
                    f"|W|_spd={snap['correctionWeightNormSpd']:.4f} "
                    f"ratio_z/spd={snap['correctionWeightNormZTemp'] / (snap['correctionWeightNormSpd'] + 1e-12):.4f}"
                )
        if getattr(model, "crtWindowAttn", False) and "gateWinAttnEntropyMean" in train_loss_stats:
            h_tr = train_loss_stats.get("gateWinAttnEntropyMean", float("nan"))
            h_tr_std = _std_from_mean_sq(h_tr, train_loss_stats.get("gateWinAttnEntropySqMean", float("nan")))
            t1_tr = train_loss_stats.get("gateWinAttnTop1Mean", float("nan"))
            t1_tr_std = _std_from_mean_sq(t1_tr, train_loss_stats.get("gateWinAttnTop1SqMean", float("nan")))
            r_tr = train_loss_stats.get("gateWinAttnTop1OverMeanMean", float("nan"))
            r_tr_std = _std_from_mean_sq(r_tr, train_loss_stats.get("gateWinAttnTop1OverMeanSqMean", float("nan")))
            h_te = test_loss_stats.get("gateWinAttnEntropyMean", float("nan"))
            h_te_std = _std_from_mean_sq(h_te, test_loss_stats.get("gateWinAttnEntropySqMean", float("nan")))
            t1_te = test_loss_stats.get("gateWinAttnTop1Mean", float("nan"))
            t1_te_std = _std_from_mean_sq(t1_te, test_loss_stats.get("gateWinAttnTop1SqMean", float("nan")))
            r_te = test_loss_stats.get("gateWinAttnTop1OverMeanMean", float("nan"))
            r_te_std = _std_from_mean_sq(r_te, test_loss_stats.get("gateWinAttnTop1OverMeanSqMean", float("nan")))
            line += (
                f" Gateα tr: H={h_tr:.4f}±{h_tr_std:.4f} max={t1_tr:.4f}±{t1_tr_std:.4f} top1/μ={r_tr:.4f}±{r_tr_std:.4f}"
                f" | te: H={h_te:.4f}±{h_te_std:.4f} max={t1_te:.4f}±{t1_te_std:.4f} top1/μ={r_te:.4f}±{r_te_std:.4f}"
            )
        print(line)

    last5_mean_test_acc = float(np.mean(epoch_test_accs[-5:]))
    last5_mean_test_auc = float(np.nanmean(np.array(epoch_test_aucs[-5:], dtype=np.float64)))
    last5_mean_test_recall = float(np.nanmean(np.array(epoch_test_recalls[-5:], dtype=np.float64)))
    last5_mean_test_spec = float(np.nanmean(np.array(epoch_test_specs[-5:], dtype=np.float64)))

    print(
        f"Fold {fold_idx_1based} summary: "
        f"best_test_acc={best_test_acc:.4f}, best_test_auc={best_test_auc:.4f}, "
        f"best_test_recall={best_test_recall:.4f}, best_test_spec={best_test_spec:.4f}, "
        f"final_test_acc={final_test_acc:.4f}, final_test_auc={final_test_auc:.4f}, "
        f"final_test_recall={final_test_recall:.4f}, final_test_spec={final_test_spec:.4f}, "
        f"last5_mean_test_acc={last5_mean_test_acc:.4f}, last5_mean_test_auc={last5_mean_test_auc:.4f}, "
        f"last5_mean_test_recall={last5_mean_test_recall:.4f}, last5_mean_test_spec={last5_mean_test_spec:.4f}"
    )

    if best_test_preds is None:
        best_test_preds = last_train_preds
        best_test_probs = last_train_probs
        best_test_labels = last_train_labels
        best_test_loss = last_train_loss

    return {
        "best_test_acc": best_test_acc,
        "best_test_auc": best_test_auc,
        "best_test_recall": best_test_recall,
        "best_test_spec": best_test_spec,
        "final_test_acc": final_test_acc,
        "final_test_auc": final_test_auc,
        "final_test_recall": final_test_recall,
        "final_test_spec": final_test_spec,
        "last5_mean_test_acc": last5_mean_test_acc,
        "last5_mean_test_auc": last5_mean_test_auc,
        "last5_mean_test_recall": last5_mean_test_recall,
        "last5_mean_test_spec": last5_mean_test_spec,
        "best_test_preds": best_test_preds,
        "best_test_probs": best_test_probs,
        "best_test_labels": best_test_labels,
        "best_test_loss": best_test_loss,
        "last_train_preds": last_train_preds,
        "last_train_probs": last_train_probs,
        "last_train_labels": last_train_labels,
        "last_train_loss": last_train_loss,
    }


def run_cgtfnet(hyperParams, datasetDetails, device="cuda:0"):
    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs
    run_seed = getattr(hyperParams, "seed", None)
    if run_seed is not None:
        run_seed = int(run_seed)

    dataset_train = getDataset(datasetDetails)
    dataset_test = getDataset(datasetDetails)
    details = Option(
        {
            "device": device,
            "nOfTrains": dataset_train.get_nOfTrains_perFold(),
            "nOfClasses": datasetDetails.nOfClasses,
            "batchSize": datasetDetails.batchSize,
            "nOfEpochs": nOfEpochs,
        }
    )

    results = []
    fold_best_accs = []
    fold_best_aucs = []
    fold_best_recalls = []
    fold_best_specs = []
    fold_final_accs = []
    fold_final_aucs = []
    fold_final_recalls = []
    fold_final_specs = []
    fold_last5_mean_accs = []
    fold_last5_mean_aucs = []
    fold_last5_mean_recalls = []
    fold_last5_mean_specs = []

    for fold in range(foldCount):
        fold_idx_1based = fold + 1
        if run_seed is not None:
            set_all_seeds(run_seed)
        else:
            set_all_seeds(datasetSeed + fold_idx_1based)

        dataset_train.setFold(fold, train=True)
        n_train_fold = len(dataset_train.targetData)
        steps_per_epoch = max(1, int(np.ceil(n_train_fold / float(datasetDetails.batchSize))))
        cosine_steps = max(1, steps_per_epoch * nOfEpochs)
        fold_details = Option({**details.dict, "cosineSchedulerSteps": cosine_steps})
        model = CGTFNetModel(hyperParams, fold_details)

        fold_res = train_and_eval_5fold_style(
            model=model,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            fold=fold,
            nOfEpochs=nOfEpochs,
            fold_idx_1based=fold_idx_1based,
        )

        train_preds = fold_res["last_train_preds"]
        train_probs = fold_res["last_train_probs"]
        train_groundTruths = fold_res["last_train_labels"]
        train_loss = fold_res["last_train_loss"]
        test_preds = fold_res["best_test_preds"]
        test_probs = fold_res["best_test_probs"]
        test_groundTruths = fold_res["best_test_labels"]
        test_loss = fold_res["best_test_loss"]

        fold_best_accs.append(fold_res["best_test_acc"])
        fold_best_aucs.append(fold_res["best_test_auc"])
        fold_best_recalls.append(fold_res["best_test_recall"])
        fold_best_specs.append(fold_res["best_test_spec"])
        fold_final_accs.append(fold_res["final_test_acc"])
        fold_final_aucs.append(fold_res["final_test_auc"])
        fold_final_recalls.append(fold_res["final_test_recall"])
        fold_final_specs.append(fold_res["final_test_spec"])
        fold_last5_mean_accs.append(fold_res["last5_mean_test_acc"])
        fold_last5_mean_aucs.append(fold_res["last5_mean_test_auc"])
        fold_last5_mean_recalls.append(fold_res["last5_mean_test_recall"])
        fold_last5_mean_specs.append(fold_res["last5_mean_test_spec"])

        results.append(
            {
                "train": {"labels": train_groundTruths, "predictions": train_preds, "probs": train_probs, "loss": train_loss},
                "test": {"labels": test_groundTruths, "predictions": test_preds, "probs": test_probs, "loss": test_loss},
            }
        )

    if foldCount == 5:
        print("\n" + "=" * 80)
        print("5-fold Summary")
        print("=" * 80)
        print(f"best_test_acc  : mean={np.mean(fold_best_accs):.4f}, std={np.std(fold_best_accs, ddof=1):.4f}, values={np.round(fold_best_accs, 4)}")
        print(f"best_test_auc  : mean={np.nanmean(fold_best_aucs):.4f}, std={np.nanstd(fold_best_aucs, ddof=1):.4f}, values={np.round(fold_best_aucs, 4)}")
        print(f"final_test_acc : mean={np.mean(fold_final_accs):.4f}, std={np.std(fold_final_accs, ddof=1):.4f}, values={np.round(fold_final_accs, 4)}")
        print(f"final_test_auc : mean={np.nanmean(fold_final_aucs):.4f}, std={np.nanstd(fold_final_aucs, ddof=1):.4f}, values={np.round(fold_final_aucs, 4)}")
        print(f"last5_mean_acc : mean={np.mean(fold_last5_mean_accs):.4f}, std={np.std(fold_last5_mean_accs, ddof=1):.4f}, values={np.round(fold_last5_mean_accs, 4)}")
        print(f"last5_mean_auc : mean={np.nanmean(fold_last5_mean_aucs):.4f}, std={np.nanstd(fold_last5_mean_aucs, ddof=1):.4f}, values={np.round(fold_last5_mean_aucs, 4)}")
        print(f"best_test_recall: mean={np.nanmean(fold_best_recalls):.4f}, std={np.nanstd(fold_best_recalls, ddof=1):.4f}, values={np.round(fold_best_recalls, 4)}")
        print(f"best_test_spec  : mean={np.nanmean(fold_best_specs):.4f}, std={np.nanstd(fold_best_specs, ddof=1):.4f}, values={np.round(fold_best_specs, 4)}")
        print(f"final_test_recall: mean={np.nanmean(fold_final_recalls):.4f}, std={np.nanstd(fold_final_recalls, ddof=1):.4f}, values={np.round(fold_final_recalls, 4)}")
        print(f"final_test_spec  : mean={np.nanmean(fold_final_specs):.4f}, std={np.nanstd(fold_final_specs, ddof=1):.4f}, values={np.round(fold_final_specs, 4)}")
        print(f"last5_mean_recall: mean={np.nanmean(fold_last5_mean_recalls):.4f}, std={np.nanstd(fold_last5_mean_recalls, ddof=1):.4f}, values={np.round(fold_last5_mean_recalls, 4)}")
        print(f"last5_mean_spec  : mean={np.nanmean(fold_last5_mean_specs):.4f}, std={np.nanstd(fold_last5_mean_specs, ddof=1):.4f}, values={np.round(fold_last5_mean_specs, 4)}")

    return results
