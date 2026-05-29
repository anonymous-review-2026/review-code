import torch
import time

from Models.CGTFNet.windowed_temporal_encoder import WindowedTemporalEncoder
from Models.Global.spdnet import BrainSPDFeaturizer, StiefelMetaOptimizer, StiefelParameter


class CGTFNetModel:
    def __init__(self, hyperParams, details):
        self.hyperParams = hyperParams
        self.details = details

        self.temporalEncoder = WindowedTemporalEncoder(hyperParams, details).to(details.device)
        self.useSpdFusion = bool(getattr(hyperParams, "useSpdFusion", False))
        self.debugModel = bool(getattr(hyperParams, "debugModel", False))
        self.debugEvery = int(getattr(hyperParams, "debugEvery", 100))
        self.debugStepLogs = bool(getattr(hyperParams, "debugStepLogs", False))
        self.globalSpdEps = float(getattr(hyperParams, "globalSpdEps", 1e-3))
        self.correctionDropout = float(getattr(hyperParams, "crtDropout", 0.3))
        self.crtWindowAttn = bool(getattr(hyperParams, "crtWindowAttn", True))
        self.crtDetachSpdQ = bool(getattr(hyperParams, "crtDetachSpdQ", True))
        self.crtAttnDim = int(getattr(hyperParams, "crtAttnDim", 32))
        self.crtAttnNum = max(1, int(getattr(hyperParams, "crtAttnNum", 1)))
        self._crt_attn_head_dim = self.crtAttnDim
        self._crt_attn_num_heads = self.crtAttnNum
        self._crt_attn_qk_total = self.crtAttnNum * self.crtAttnDim
        self._stepCount = 0

        temporal_dim = int(hyperParams.dim)
        if self.useSpdFusion:
            self.connectivityEncoder = BrainSPDFeaturizer(
                input_rois=temporal_dim,
                hidden_sizes=(64, 32),
                epsilon=float(self.globalSpdEps),
            ).to(details.device)
            spd_feat_dim = int(self.connectivityEncoder.out_dim)
            self._spd_feat_dim = spd_feat_dim
            d_total = self._crt_attn_qk_total
            if self.crtWindowAttn:
                self.connectivityQuery = torch.nn.Linear(spd_feat_dim, d_total).to(details.device)
                self.temporalKey = torch.nn.Linear(temporal_dim, d_total).to(details.device)
                corr_in = spd_feat_dim + temporal_dim
                self.classifier = torch.nn.Linear(corr_in, details.nOfClasses).to(details.device)
            else:
                self.connectivityQuery = None
                self.temporalKey = None
                self.classifier = torch.nn.Linear(temporal_dim + spd_feat_dim, details.nOfClasses).to(details.device)
            self.classifierDropout = torch.nn.Dropout(p=self.correctionDropout)
        else:
            self.connectivityEncoder = None
            self._spd_feat_dim = None
            self.connectivityQuery = None
            self.temporalKey = None
            self.classifier = None
            self.classifierDropout = None

        ls = float(getattr(hyperParams, "labelSmoothing", 0.0))
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)

        all_params = list(self.temporalEncoder.parameters())
        fusion_param_ids = set()
        if self.connectivityEncoder is not None:
            all_params += list(self.connectivityEncoder.parameters())
            for p in self.classifier.parameters():
                all_params.append(p)
                fusion_param_ids.add(id(p))
            if self.connectivityQuery is not None:
                for p in self.connectivityQuery.parameters():
                    all_params.append(p)
                    fusion_param_ids.add(id(p))
                for p in self.temporalKey.parameters():
                    all_params.append(p)
                    fusion_param_ids.add(id(p))

        wd_base = float(hyperParams.weightDecay)
        _crt_wd = getattr(hyperParams, "crtWeightDecay", None)
        wd_fusion = wd_base if _crt_wd is None else float(_crt_wd)

        stiefel_params = []
        euclidean_base = []
        euclidean_fusion = []
        for param in all_params:
            if isinstance(param, StiefelParameter):
                stiefel_params.append(param)
            elif id(param) in fusion_param_ids:
                euclidean_fusion.append(param)
            else:
                euclidean_base.append(param)

        param_groups = []
        if stiefel_params:
            param_groups.append({"params": stiefel_params, "weight_decay": 0.0})
        if euclidean_base:
            param_groups.append({"params": euclidean_base, "weight_decay": wd_base})
        if euclidean_fusion:
            param_groups.append({"params": euclidean_fusion, "weight_decay": wd_fusion})

        self.baseOptimizer = torch.optim.Adam(param_groups, lr=hyperParams.lr)
        self.optimizer = StiefelMetaOptimizer(self.baseOptimizer)
        sched = str(getattr(hyperParams, "lrScheduler", "none")).lower()
        total_steps = int(getattr(details, "cosineSchedulerSteps", 0))
        eta_min = float(getattr(hyperParams, "minLr", 0.0))
        if sched == "cosine" and total_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.baseOptimizer, T_max=total_steps, eta_min=eta_min)
        else:
            self.scheduler = None

    def step(self, x, y, train=True, oasCorr=None):
        self._stepCount += 1
        inputs, y, oasCorr = self.prepareInput(x, y, oasCorr)

        if train:
            self.temporalEncoder.train()
            if self.connectivityEncoder is not None:
                self.connectivityEncoder.train()
                self.classifier.train()
                self.classifierDropout.train()
                if self.connectivityQuery is not None:
                    self.connectivityQuery.train()
                    self.temporalKey.train()
        else:
            self.temporalEncoder.eval()
            if self.connectivityEncoder is not None:
                self.connectivityEncoder.eval()
                self.classifier.eval()
                self.classifierDropout.eval()
                if self.connectivityQuery is not None:
                    self.connectivityQuery.eval()
                    self.temporalKey.eval()

        temporal_logits, cls = self.temporalEncoder(*inputs)
        temporal_feat = cls.mean(dim=1)

        spd_feat = None
        win_attn = None
        if self.useSpdFusion:
            if oasCorr is None:
                raise ValueError("SPD fusion requires precomputed 'oasCorr' for every sample.")
            spd_feat = self.connectivityEncoder(oasCorr)
            if self.crtWindowAttn and self.connectivityQuery is not None:
                d_h = self._crt_attn_head_dim
                n_h = self._crt_attn_num_heads
                scale = d_h ** 0.5
                spd_for_q = spd_feat.detach() if self.crtDetachSpdQ else spd_feat
                q = self.connectivityQuery(spd_for_q)
                k = self.temporalKey(cls)
                bsz, n_win, _ = k.shape
                q = q.view(bsz, n_h, d_h)
                k = k.view(bsz, n_win, n_h, d_h)
                scores_h = (q.unsqueeze(1) * k).sum(dim=-1) / scale
                scores = scores_h.mean(dim=-1)
                attn = torch.softmax(scores, dim=1)
                win_attn = attn
                z_temp = (attn.unsqueeze(-1) * cls).sum(dim=1)
                fusion_in = torch.cat([spd_feat, z_temp], dim=1)
                fusion_in = self.classifierDropout(fusion_in)
                logits = self.classifier(fusion_in)
            else:
                fusion_in = torch.cat([temporal_feat, spd_feat], dim=1)
                fusion_in = self.classifierDropout(fusion_in)
                logits = self.classifier(fusion_in)
        else:
            logits = temporal_logits

        loss, loss_info = self.getLoss(logits, y, cls, fusion_logits=logits if self.useSpdFusion else None, spd_feat=spd_feat, win_attn=win_attn)
        preds = logits.argmax(1)
        probs = logits.softmax(1)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        loss = loss.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")
        normalized_loss_info = {}
        for key, value in loss_info.items():
            if isinstance(value, torch.Tensor):
                normalized_loss_info[key] = float(value.detach().to("cpu").item())
            else:
                normalized_loss_info[key] = float(value)
        y = y.to("cpu")
        torch.cuda.empty_cache()
        return loss, preds, probs, y, normalized_loss_info

    def prepareInput(self, x, y, oasCorr=None):
        x = x.to(self.details.device)
        y = y.to(self.details.device)
        if oasCorr is not None:
            if not isinstance(oasCorr, torch.Tensor):
                oasCorr = torch.as_tensor(oasCorr)
            oasCorr = oasCorr.to(self.details.device)
        return (x,), y, oasCorr

    def getLoss(self, yHat, y, cls, fusion_logits=None, spd_feat=None, win_attn=None):
        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))
        final_loss = self.criterion(yHat, y)
        weighted_cons_loss = clsLoss * self.hyperParams.lambdaCons
        total_loss = final_loss + weighted_cons_loss

        if fusion_logits is not None:
            fusion_mean = fusion_logits.mean()
            fusion_var = fusion_logits.var(unbiased=False)
            fusion_norm = fusion_logits.norm(dim=1).mean()
            if spd_feat is not None:
                spd_feat_norm = spd_feat.norm(dim=1).mean()
                fusion_norm_to_spd_feat = fusion_norm / (spd_feat_norm + 1e-12)
            else:
                fusion_norm_to_spd_feat = fusion_norm.new_zeros(())
            with torch.no_grad():
                probs_f = torch.softmax(fusion_logits, dim=1)
                eps = 1e-8
                fusion_pred_conf_mean = probs_f.max(dim=1).values.mean()
                fusion_pred_entropy_mean = (-(probs_f * (probs_f + eps).log()).sum(dim=1)).mean()
                if fusion_logits.shape[1] == 2:
                    fusion_logit_margin_mean = (fusion_logits[:, 1] - fusion_logits[:, 0]).abs().mean()
                else:
                    fusion_logit_margin_mean = fusion_logits.new_zeros(())
        else:
            fusion_mean = final_loss.new_zeros(())
            fusion_var = final_loss.new_zeros(())
            fusion_norm = final_loss.new_zeros(())
            fusion_norm_to_spd_feat = final_loss.new_zeros(())
            fusion_pred_conf_mean = final_loss.new_zeros(())
            fusion_pred_entropy_mean = final_loss.new_zeros(())
            fusion_logit_margin_mean = final_loss.new_zeros(())

        loss_info = {
            "totalLoss": total_loss,
            "finalLossWeighted": final_loss,
            "consistencyLossWeighted": weighted_cons_loss,
            "sampleCount": y.shape[0],
            "fusionLogitMean": fusion_mean,
            "fusionLogitVar": fusion_var,
            "fusionLogitNorm": fusion_norm,
            "fusionLogitNormToSpdFeatNorm": fusion_norm_to_spd_feat,
            "fusionPredConfMean": fusion_pred_conf_mean,
            "fusionPredEntropyMean": fusion_pred_entropy_mean,
            "fusionLogitMarginMean": fusion_logit_margin_mean,
        }
        if win_attn is not None:
            with torch.no_grad():
                eps = 1e-8
                ent = -(win_attn * (win_attn + eps).log()).sum(dim=1)
                top1 = win_attn.max(dim=1).values
                mean_w = win_attn.mean(dim=1)
                ratio_tm = top1 / (mean_w + eps)
                loss_info["gateWinAttnEntropyMean"] = ent.mean()
                loss_info["gateWinAttnEntropySqMean"] = (ent * ent).mean()
                loss_info["gateWinAttnTop1Mean"] = top1.mean()
                loss_info["gateWinAttnTop1SqMean"] = (top1 * top1).mean()
                loss_info["gateWinAttnTop1OverMeanMean"] = ratio_tm.mean()
                loss_info["gateWinAttnTop1OverMeanSqMean"] = (ratio_tm * ratio_tm).mean()
        return total_loss, loss_info

    def set_epoch(self, epoch: int):
        return None

    def get_debug_snapshot(self):
        snapshot = {"stepCount": self._stepCount}
        if self.useSpdFusion and self.classifier is not None:
            w = self.classifier.weight.detach()
            if self.crtWindowAttn and self.connectivityQuery is not None:
                sdim = int(self._spd_feat_dim)
                z_dim = int(self.hyperParams.dim)
                ztemp_w = w[:, :z_dim]
                spd_w = w[:, z_dim : z_dim + sdim]
                snapshot["correctionWeightNormZTemp"] = float(ztemp_w.norm().item())
                snapshot["correctionWeightNormSpd"] = float(spd_w.norm().item())
            else:
                temporal_dim = int(self.hyperParams.dim)
                temporal_w = w[:, :temporal_dim]
                spd_w = w[:, temporal_dim:]
                snapshot["correctionWeightNormBolt"] = float(temporal_w.norm().item())
                snapshot["correctionWeightNormSpd"] = float(spd_w.norm().item())
        return snapshot
