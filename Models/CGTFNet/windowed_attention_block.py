import torch
from torch import nn
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_

from Models.CGTFNet.temporal_window_ops import build_sliding_windows


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, window_size, receptive_size, num_heads, head_dim=20, attention_bias=True, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.windowSize = window_size
        self.receptiveSize = receptive_size
        self.numHeads = num_heads
        self.scale = head_dim ** -0.5
        self.attentionBias = attention_bias

        max_disparity = window_size - 1 + (receptive_size - window_size) // 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * max_disparity + 1, num_heads))
        self.cls_bias_sequence_up = nn.Parameter(torch.zeros((1, num_heads, 1, receptive_size)))
        self.cls_bias_sequence_down = nn.Parameter(torch.zeros(1, num_heads, window_size, 1))
        self.cls_bias_self = nn.Parameter(torch.zeros((1, num_heads, 1, 1)))

        coords_x = torch.arange(self.windowSize)
        coords_x_ = torch.arange(self.receptiveSize) - (self.receptiveSize - self.windowSize) // 2
        relative_coords = coords_x[:, None] - coords_x_[None, :]
        relative_coords[:, :] += max_disparity
        self.register_buffer("relative_position_index", relative_coords)

        self.q = nn.Linear(dim, head_dim * num_heads, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * head_dim * num_heads, bias=qkv_bias)
        self.attnDrop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.projDrop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        trunc_normal_(self.cls_bias_sequence_up, std=0.02)
        trunc_normal_(self.cls_bias_sequence_down, std=0.02)
        trunc_normal_(self.cls_bias_self, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_, mask, nW):
        B_, N, _ = x.shape
        _, M, _ = x_.shape
        N = N - 1
        M = M - 1
        B = B_ // nW
        mask_left, mask_right = mask

        q = self.q(x)
        k, v = self.kv(x_).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.numHeads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.numHeads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.numHeads)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, M, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        if self.attentionBias:
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + relative_position_bias.unsqueeze(0)
            attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self
            attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up
            attn[:, :, 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down

        mask_left = repeat(mask_left, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)
        mask_right = repeat(mask_right, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)
        mask_value = max_neg_value(attn)
        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", nW=nW)
        maskCount = min(mask_left.shape[0], attn.shape[1])
        mask_left = mask_left[:, :maskCount]
        mask_right = mask_right[:, -maskCount:]
        attn[:, :maskCount].masked_fill_(mask_left == 1, mask_value)
        attn[:, -maskCount:].masked_fill_(mask_right == 1, mask_value)
        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")
        attn = self.softmax(attn)
        attn = self.attnDrop(attn)
        x = torch.matmul(attn, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.projDrop(x)
        return x


class LocalWindowTransformer(nn.Module):
    def __init__(self, dim, window_size, shift_size, receptive_size, num_heads, head_dim, mlp_ratio, attention_bias, drop, attn_drop):
        super().__init__()
        self.attention = WindowSelfAttention(
            dim=dim,
            window_size=window_size,
            receptive_size=receptive_size,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mlp = FeedForward(dim=dim, mult=mlp_ratio, dropout=drop)
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.shiftSize = shift_size

    def forward(self, x, cls, windowX, windowX_, mask, nW):
        windowXTrans = self.attention(self.attn_norm(windowX), self.attn_norm(windowX_), mask, nW)
        clsTrans = windowXTrans[:, :1]
        xTrans = windowXTrans[:, 1:]
        clsTrans = rearrange(clsTrans, "(b nW) l c -> b (nW l) c", nW=nW)
        xTrans = rearrange(xTrans, "(b nW) l c -> b nW l c", nW=nW)
        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)
        clsTrans = clsTrans + cls
        xTrans = xTrans + x
        xTrans = xTrans + self.mlp(self.mlp_norm(xTrans))
        clsTrans = clsTrans + self.mlp(self.mlp_norm(clsTrans))
        return xTrans, clsTrans

    def gatherWindows(self, windowedX, dynamicLength, shiftSize):
        batchSize = windowedX.shape[0]
        windowLength = windowedX.shape[2]
        nW = windowedX.shape[1]
        C = windowedX.shape[-1]
        device = windowedX.device

        destination = torch.zeros((batchSize, dynamicLength, C)).to(device)
        scalerDestination = torch.zeros((batchSize, dynamicLength, C)).to(device)
        indexes = torch.tensor([[j + (i * shiftSize) for j in range(windowLength)] for i in range(nW)]).to(device)
        indexes = indexes[None, :, :, None].repeat((batchSize, 1, 1, C))
        src = rearrange(windowedX, "b n w c -> b (n w) c")
        indexes = rearrange(indexes, "b n w c -> b (n w) c")
        destination.scatter_add_(dim=1, index=indexes, src=src)
        scalerSrc = torch.ones((windowLength)).to(device)[None, None, :, None].repeat(batchSize, nW, 1, C)
        scalerSrc = rearrange(scalerSrc, "b n w c -> b (n w) c")
        scalerDestination.scatter_add_(dim=1, index=indexes, src=scalerSrc)
        destination = destination / scalerDestination
        return destination


class WindowedTemporalBlock(nn.Module):
    def __init__(self, dim, numHeads, headDim, windowSize, receptiveSize, shiftSize, mlpRatio=1.0, drop=0.0, attnDrop=0.0, attentionBias=True):
        assert (receptiveSize - windowSize) % 2 == 0
        super().__init__()
        self.transformer = LocalWindowTransformer(
            dim=dim,
            window_size=windowSize,
            shift_size=shiftSize,
            receptive_size=receptiveSize,
            num_heads=numHeads,
            head_dim=headDim,
            mlp_ratio=mlpRatio,
            attention_bias=attentionBias,
            drop=drop,
            attn_drop=attnDrop,
        )
        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize
        self.remainder = (self.receptiveSize - self.windowSize) // 2

        maskCount = self.remainder // shiftSize + 1
        mask_left = torch.zeros(maskCount, self.windowSize + 1, self.receptiveSize + 1)
        mask_right = torch.zeros(maskCount, self.windowSize + 1, self.receptiveSize + 1)
        for i in range(maskCount):
            if self.remainder > 0:
                mask_left[i, :, 1 : 1 + self.remainder - shiftSize * i] = 1
                if -self.remainder + shiftSize * i > 0:
                    mask_right[maskCount - 1 - i, :, -self.remainder + shiftSize * i :] = 1
        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)

    def forward(self, x, cls):
        B, _, C = x.shape
        device = x.device
        Z = self.windowSize + self.shiftSize * (cls.shape[1] - 1)
        x = x[:, :Z]

        x_ = torch.cat([torch.zeros((B, self.remainder, C), device=device), x, torch.zeros((B, self.remainder, C), device=device)], dim=1)
        windowedX, _ = build_sliding_windows(x.transpose(2, 1), self.windowSize, self.shiftSize)
        windowedX = windowedX.transpose(2, 3)
        windowedX_, _ = build_sliding_windows(x_.transpose(2, 1), self.receptiveSize, self.shiftSize)
        windowedX_ = windowedX_.transpose(2, 3)
        nW = windowedX.shape[1]

        xcls = torch.cat([cls.unsqueeze(dim=2), windowedX], dim=2)
        xcls = rearrange(xcls, "b nw l c -> (b nw) l c")
        xcls_ = torch.cat([cls.unsqueeze(dim=2), windowedX_], dim=2)
        xcls_ = rearrange(xcls_, "b nw l c -> (b nw) l c")
        masks = [self.mask_left, self.mask_right]
        return self.transformer(x, cls, xcls, xcls_, masks, nW)
