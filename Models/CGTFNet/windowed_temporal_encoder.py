import torch
from torch import nn
import numpy as np

from Models.CGTFNet.windowed_attention_block import WindowedTemporalBlock


class WindowedTemporalEncoder(nn.Module):
    def __init__(self, hyperParams, details):
        super().__init__()

        dim = hyperParams.dim
        nOfClasses = details.nOfClasses
        self.hyperParams = hyperParams

        self.inputNorm = nn.LayerNorm(dim)
        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = []

        legacy_shift = int(hyperParams.windowSize * getattr(hyperParams, "shiftCoeff", 2.0 / 5.0))
        shiftSize = int(getattr(hyperParams, "shiftSize", legacy_shift))
        self.shiftSize = shiftSize
        baseFringe = int(getattr(hyperParams, "fringeSize", 0))
        fringeGrowth = int(getattr(hyperParams, "fringeCoeff", 0))
        if shiftSize <= 0:
            raise ValueError("shiftSize must be positive.")
        if baseFringe < 0 or fringeGrowth < 0:
            raise ValueError("fringeSize and fringeCoeff must be non-negative.")
        self.fringeSizes = []
        self.receptiveSizes = []

        for i, _ in enumerate(range(hyperParams.nOfLayers)):
            fringeSize = baseFringe + i * fringeGrowth
            receptiveSize = hyperParams.windowSize + 2 * fringeSize
            print(
                "layer {} : windowSize={}, shiftSize={}, fringeSize={}, receptiveSize={}".format(
                    i, hyperParams.windowSize, shiftSize, fringeSize, receptiveSize
                )
            )
            self.fringeSizes.append(fringeSize)
            self.receptiveSizes.append(receptiveSize)
            self.blocks.append(
                WindowedTemporalBlock(
                    dim=hyperParams.dim,
                    numHeads=hyperParams.numHeads,
                    headDim=hyperParams.headDim,
                    windowSize=hyperParams.windowSize,
                    receptiveSize=receptiveSize,
                    shiftSize=shiftSize,
                    mlpRatio=hyperParams.mlpRatio,
                    attentionBias=hyperParams.attentionBias,
                    drop=hyperParams.drop,
                    attnDrop=hyperParams.attnDrop,
                )
            )

        self.blocks = nn.ModuleList(self.blocks)
        self.encoder_postNorm = nn.LayerNorm(dim)
        self.classifierHead = nn.Linear(dim, nOfClasses)
        self.last_numberOfWindows = None
        self.initializeWeights()

    def initializeWeights(self):
        torch.nn.init.normal_(self.clsToken, std=1.0)

    def calculateFlops(self, T):
        windowSize = self.hyperParams.windowSize
        shiftSize = self.shiftSize
        focalSizes = self.focalSizes
        macs = []
        nW = (T - windowSize) // shiftSize + 1
        C = 400
        H = self.hyperParams.numHeads
        D = self.hyperParams.headDim

        for _, focalSize in enumerate(focalSizes):
            mac = 0
            mac += nW * (1 + windowSize) * C * H * D * 3
            mac += 2 * nW * H * D * (1 + windowSize) * (1 + focalSize)
            mac += nW * (1 + windowSize) * C * H * D
            mac += 2 * (T + nW) * C * C
            macs.append(mac)

        return macs, np.sum(macs) * 2

    def forward(self, roiSignals):
        roiSignals = roiSignals.permute((0, 2, 1))
        batchSize = roiSignals.shape[0]
        T = roiSignals.shape[1]
        nW = (T - self.hyperParams.windowSize) // self.shiftSize + 1
        cls = self.clsToken.repeat(batchSize, nW, 1)
        self.last_numberOfWindows = nW

        for block in self.blocks:
            roiSignals, cls = block(roiSignals, cls)

        cls = self.encoder_postNorm(cls)
        if self.hyperParams.pooling == "cls":
            logits = self.classifierHead(cls.mean(dim=1))
        elif self.hyperParams.pooling == "gmp":
            logits = self.classifierHead(roiSignals.mean(dim=1))

        torch.cuda.empty_cache()
        return logits, cls
