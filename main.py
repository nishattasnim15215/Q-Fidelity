import os
import time
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from sklearn.metrics import roc_auc_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class SampleItem:
    image_path: str
    mask_path: Optional[str]
    label: int
    defect: str
    split: str


_GLOBAL_QUAL_ROWS: List[Dict[str, object]] = []


# =============================================================================
# Reproducibility and paths
# =============================================================================
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_paths(base_dir: str, category: str, seed: Optional[int] = None) -> Dict[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    sub = category if seed is None else os.path.join(category, f"seed_{seed}")
    cat_dir = os.path.join(base_dir, sub)
    os.makedirs(cat_dir, exist_ok=True)

    prefix = category if seed is None else f"{category}_seed_{seed}"
    return {
        "base": base_dir,
        "cat": cat_dir,
        "csv_summary": os.path.join(cat_dir, f"{prefix}_summary.csv"),
        "csv_predictions": os.path.join(cat_dir, f"{prefix}_predictions.csv"),
        "csv_speed": os.path.join(cat_dir, f"{prefix}_speed.csv"),
        "csv_ablation": os.path.join(cat_dir, f"{prefix}_ablation.csv"),
    }


# =============================================================================
# Dataset indexing
# =============================================================================
def list_images(root_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    out: List[str] = []

    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(exts):
                out.append(os.path.join(root, name))

    out.sort()
    return out


def index_mvtec_ad2(dataset_root: str, category: str) -> Dict[str, List[SampleItem]]:
    category_root = os.path.join(dataset_root, category)
    splits = ["train", "validation", "test_public", "test_private", "test_private_mixed"]
    index: Dict[str, List[SampleItem]] = {s: [] for s in splits}

    if not os.path.isdir(category_root):
        raise FileNotFoundError(f"Category folder not found: {category_root}")

    for split in splits:
        split_root = os.path.join(category_root, split)
        if not os.path.isdir(split_root):
            continue

        subdirs = [d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))]

        if len(subdirs) == 0:
            imgs = list_images(split_root)
            for path in imgs:
                index[split].append(SampleItem(path, None, 0, "good", split))
            continue

        for defect in subdirs:
            defect_root = os.path.join(split_root, defect)
            imgs = list_images(defect_root)

            for img_path in imgs:
                label = 0 if defect.lower() == "good" else 1
                mask_path = None

                if split == "test_public" and label == 1:
                    mp = img_path.replace(
                        os.sep + "bad" + os.sep,
                        os.sep + "ground_truth" + os.sep + "bad" + os.sep,
                    )
                    if os.path.exists(mp):
                        mask_path = mp
                    else:
                        mp2 = mp.replace(".png", "_mask.png")
                        if os.path.exists(mp2):
                            mask_path = mp2

                index[split].append(SampleItem(img_path, mask_path, label, defect, split))

    for split in splits:
        index[split].sort(key=lambda x: x.image_path)

    return index


def parse_csv_set(value: str) -> Set[str]:
    out: Set[str] = set()
    for tok in (value or "").split(","):
        tok = tok.strip()
        if tok:
            out.add(tok)
    return out


def parse_csv_ints(value: str) -> List[int]:
    vals: List[int] = []
    for tok in (value or "").split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals if vals else [7]


# =============================================================================
# Dataset
# =============================================================================
class MVTecAD2Dataset(Dataset):
    def __init__(
        self,
        items: List[SampleItem],
        img_size: int,
        category: str,
        preprocess_categories: Set[str],
    ):
        self.items = items
        self.img_size = img_size
        self.category = category
        self.use_preprocess_tune = category in preprocess_categories

    def __len__(self) -> int:
        return len(self.items)

    def _prep_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        x = x.float().div(255.0)
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.squeeze(0).contiguous()

        if not self.use_preprocess_tune:
            return x

        eps = 1e-6

        ch_mean = x.mean(dim=(1, 2), keepdim=True)
        gray_mean = ch_mean.mean(dim=0, keepdim=True)
        scale = gray_mean / (ch_mean + eps)
        x = (x * scale).clamp(0.0, 1.0)

        flat = x.reshape(3, -1)
        lo = torch.quantile(flat, 0.01, dim=1).view(3, 1, 1)
        hi = torch.quantile(flat, 0.99, dim=1).view(3, 1, 1)
        x = (x - lo) / (hi - lo + eps)
        x = x.clamp(0.0, 1.0)

        lum = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).mean().item()
        if lum < 0.35:
            gamma = 0.80
        elif lum > 0.65:
            gamma = 1.15
        else:
            gamma = 1.0

        if gamma != 1.0:
            x = torch.pow(x.clamp(0.0, 1.0), gamma)

        return x

    def _load_image(self, path: str) -> torch.Tensor:
        x = read_image(path)
        return self._prep_rgb(x)

    def _load_mask(self, path: Optional[str]) -> torch.Tensor:
        if path is None or not os.path.exists(path):
            return torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        m = read_image(path)
        if m.ndim == 3 and m.shape[0] > 1:
            m = m[:1]
        if m.ndim == 2:
            m = m.unsqueeze(0)

        m = (m.float() > 0).float()
        m = m.unsqueeze(0)
        m = F.interpolate(m, size=(self.img_size, self.img_size), mode="nearest")
        return m.squeeze(0).contiguous()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        image = self._load_image(item.image_path)
        mask = self._load_mask(item.mask_path)
        label = torch.tensor(item.label, dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": item.image_path,
            "defect": item.defect,
            "split": item.split,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    paths = [b["path"] for b in batch]
    defects = [b["defect"] for b in batch]
    splits = [b["split"] for b in batch]

    return {
        "image": images,
        "mask": masks,
        "label": labels,
        "path": paths,
        "defect": defects,
        "split": splits,
    }


def log_data_stats(index: Dict[str, List[SampleItem]], category: str) -> None:
    counts = {k: len(v) for k, v in index.items()}
    print(f"\nDataset summary | category={category}")
    print(json.dumps(counts, indent=2))


# =============================================================================
# Public metrics
# =============================================================================
def roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def best_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    ths = np.linspace(y_score.min(), y_score.max(), num_thresholds)
    best_f1 = -1.0
    best_th = ths[0]

    for th in ths:
        pred = (y_score >= th).astype(np.int32)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return float(best_th), float(best_f1)


def eval_public_metrics(
    labels: np.ndarray,
    img_scores: np.ndarray,
    masks: np.ndarray,
    pix_scores: np.ndarray,
) -> Dict[str, float]:
    img_auc = roc_auc_safe(labels, img_scores)
    m_flat = masks.reshape(-1).astype(np.uint8)
    s_flat = pix_scores.reshape(-1).astype(np.float32)
    pix_auc = roc_auc_safe(m_flat, s_flat)

    if len(np.unique(labels)) < 2:
        f1_pct = float("nan")
    else:
        _, f1_val = best_f1_threshold(
            labels.astype(np.int32),
            img_scores.astype(np.float32),
            num_thresholds=200,
        )
        f1_pct = float(f1_val * 100.0)

    return {
        "ImageAUROC": float(img_auc),
        "PixelAUROC": float(pix_auc),
        "ClassF1_public_pct": float(f1_pct),
    }


def count_params_m(model: nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e6


def max_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device=device) / (1024.0 ** 2))


# =============================================================================
# Shared modules
# =============================================================================
class PatchProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, seed: int, mode: str = "frozen_random"):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        w = torch.randn(out_dim, in_dim, generator=g) / math.sqrt(in_dim)

        self.mode = mode
        if mode == "frozen_random":
            self.register_buffer("weight", w, persistent=True)
            self.bias = None
        elif mode == "trainable_linear":
            self.linear = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                self.linear.weight.copy_(w)
                self.linear.bias.zero_()
        else:
            raise ValueError(f"Unsupported projection_mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "frozen_random":
            return F.linear(x, self.weight)
        return self.linear(x)


class IlluminationEncoder(nn.Module):
    def __init__(self, mode: str = "stats", hidden_dim: int = 32):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.net = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(1, 2, 3))
        sd = x.std(dim=(1, 2, 3)) + 1e-6
        ch_mean = x.mean(dim=(2, 3))
        feats = torch.cat([mu[:, None], sd[:, None], ch_mean], dim=1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]

        if self.mode == "none":
            return torch.zeros((b, 2), dtype=x.dtype, device=x.device)

        feats = self._features(x)
        if self.mode == "stats":
            return torch.tanh(feats[:, :2] * 3.0)
        if self.mode == "learned":
            return torch.tanh(self.net(feats))

        raise ValueError(f"Unsupported illumination_token_mode: {self.mode}")


# =============================================================================
# Quantum simulator branch
# =============================================================================
def make_cnot_perm(n_qubits: int, control: int, target: int) -> torch.Tensor:
    dim = 1 << n_qubits
    idx = torch.arange(dim, dtype=torch.long)
    c_mask = 1 << control
    t_mask = 1 << target
    control_on = (idx & c_mask) != 0
    flipped = idx ^ t_mask
    return torch.where(control_on, flipped, idx)


def apply_ry(state: torch.Tensor, theta: torch.Tensor, wire: int, n_qubits: int) -> torch.Tensor:
    bsz = state.shape[0]
    dim = 1 << n_qubits
    s = state.view(bsz, 1 << wire, 2, 1 << (n_qubits - wire - 1))
    c = torch.cos(theta * 0.5).to(state.dtype).view(bsz, 1, 1, 1)
    si = torch.sin(theta * 0.5).to(state.dtype).view(bsz, 1, 1, 1)
    a0 = s[:, :, 0:1, :]
    a1 = s[:, :, 1:2, :]
    y0 = c * a0 - si * a1
    y1 = si * a0 + c * a1
    return torch.cat([y0, y1], dim=2).view(bsz, dim)


def apply_rz(state: torch.Tensor, theta: torch.Tensor, wire: int, n_qubits: int) -> torch.Tensor:
    bsz = state.shape[0]
    dim = 1 << n_qubits
    s = state.view(bsz, 1 << wire, 2, 1 << (n_qubits - wire - 1))
    phase0 = torch.exp((-0.5j) * theta.to(torch.float32)).to(state.dtype).view(bsz, 1, 1, 1)
    phase1 = torch.exp((0.5j) * theta.to(torch.float32)).to(state.dtype).view(bsz, 1, 1, 1)
    s0 = s[:, :, 0:1, :] * phase0
    s1 = s[:, :, 1:2, :] * phase1
    return torch.cat([s0, s1], dim=2).view(bsz, dim)


def apply_cnot(state: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return state.index_select(dim=1, index=perm)


class QuantumSimulatorBranch(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_qubits: int,
        n_layers: int,
        seed: int,
        projection_mode: str,
        illumination_token_mode: str,
        noise_mode: str,
        noise_scale: float,
    ):
        super().__init__()
        self.branch_name = "quantum_simulator"
        self.backend_name = "torch_statevector_simulator"
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.use_noise = noise_mode != "none"
        self.use_entangle = True
        self.active_layers = n_layers

        patch_dim = patch_size * patch_size * 3
        out_dim = 2 * n_qubits

        self.projector = PatchProjector(patch_dim, out_dim, seed=seed, mode=projection_mode)
        self.illumination = IlluminationEncoder(mode=illumination_token_mode)
        self.noise_mlp = (
            nn.Sequential(
                nn.Linear(2 + out_dim, 64),
                nn.GELU(),
                nn.Linear(64, 2 * n_qubits),
            )
            if noise_mode == "learned"
            else None
        )

        theta = torch.zeros(n_layers, n_qubits, 2, dtype=torch.float32)
        nn.init.uniform_(theta, a=-0.05, b=0.05)
        self.theta = nn.Parameter(theta)

        perms = []
        for q in range(n_qubits - 1):
            perms.append(make_cnot_perm(n_qubits, q, q + 1))
        perms.append(make_cnot_perm(n_qubits, n_qubits - 1, 0))
        self.register_buffer("cnot_perms", torch.stack(perms, dim=0), persistent=True)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = F.unfold(x, kernel_size=(p, p), stride=(p, p))
        return patches.transpose(1, 2).contiguous()

    def encode_angles(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        flat = patches.reshape(b * n, d)
        angles = torch.tanh(self.projector(flat)) * math.pi
        return angles.view(b, n, -1)

    def _init_state(self, bsz: int, device: torch.device) -> torch.Tensor:
        dim = 1 << self.n_qubits
        st = torch.zeros((bsz, dim), dtype=torch.complex64, device=device)
        st[:, 0] = 1.0 + 0.0j
        return st

    def _apply_fixed(self, state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        for q in range(self.n_qubits):
            state = apply_ry(state, angles[:, 2 * q + 0], q, self.n_qubits)
            state = apply_rz(state, angles[:, 2 * q + 1], q, self.n_qubits)
        return state

    def _build_noise(self, illum_tok: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        if not self.use_noise or self.noise_mode == "none":
            return torch.zeros((angles.shape[0], 2 * self.n_qubits), dtype=angles.dtype, device=angles.device)

        mu = illum_tok[:, 0:1].to(torch.float32)
        sd = illum_tok[:, 1:2].to(torch.float32)
        mix = (angles.to(torch.float32) * 0.13).sum(dim=1, keepdim=True)

        if self.noise_mode == "sinusoidal":
            base = (0.7 * mu + 0.3 * sd) * self.noise_scale
            val = torch.sin(mix + base) * math.pi * 0.5
            outs = []
            for q in range(self.n_qubits):
                outs.append(val[:, 0] * (1.0 + 0.07 * (q + 1)))
                outs.append(val[:, 0] * (1.0 - 0.05 * (q + 1)))
            return torch.stack(outs, dim=1)

        if self.noise_mode == "linear":
            base = (0.5 * mu + 0.5 * sd) * self.noise_scale
            outs = []
            for q in range(self.n_qubits):
                outs.append((mix[:, 0] + base[:, 0]) * (0.10 + 0.02 * q))
                outs.append((mix[:, 0] - base[:, 0]) * (0.08 + 0.015 * q))
            return torch.stack(outs, dim=1)

        if self.noise_mode == "learned":
            h = torch.cat([illum_tok.to(torch.float32), angles.to(torch.float32)], dim=1)
            return torch.tanh(self.noise_mlp(h)) * math.pi * self.noise_scale

        raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

    def _apply_noise(self, state: torch.Tensor, noise_angles: torch.Tensor) -> torch.Tensor:
        if not self.use_noise:
            return state

        for q in range(self.n_qubits):
            state = apply_ry(state, noise_angles[:, 2 * q + 0], q, self.n_qubits)
            state = apply_rz(state, noise_angles[:, 2 * q + 1], q, self.n_qubits)

        return state

    def _apply_trainable(self, state: torch.Tensor) -> torch.Tensor:
        bsz = state.shape[0]
        layers = int(max(0, min(self.n_layers, self.active_layers)))

        for l in range(layers):
            for q in range(self.n_qubits):
                state = apply_ry(state, self.theta[l, q, 0].expand(bsz), q, self.n_qubits)
                state = apply_rz(state, self.theta[l, q, 1].expand(bsz), q, self.n_qubits)

            if self.use_entangle:
                for k in range(self.cnot_perms.shape[0]):
                    state = apply_cnot(state, self.cnot_perms[k])

        return state

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b = x.shape[0]
        illum = self.illumination(x)
        patches = self.patchify(x)
        angles = self.encode_angles(patches)
        _, n_patches, _ = angles.shape

        angles_flat = angles.view(b * n_patches, -1)
        illum_rep = illum.unsqueeze(1).expand(b, n_patches, 2).reshape(b * n_patches, 2)

        clean = self._apply_fixed(self._init_state(b * n_patches, x.device), angles_flat)
        noisy = self._apply_fixed(self._init_state(b * n_patches, x.device), angles_flat)
        noise_angles = self._build_noise(illum_rep, angles_flat)
        noisy = self._apply_noise(noisy, noise_angles)
        out = self._apply_trainable(noisy)

        overlap = (clean.conj() * out).sum(dim=1)
        fidelity = (overlap.abs() ** 2).clamp(0.0, 1.0)
        patch_score = (1.0 - fidelity).view(b, n_patches)
        img_score = patch_score.mean(dim=1)

        aux = {
            "fidelity": fidelity.view(b, n_patches),
            "illumination_token": illum,
        }
        return img_score, patch_score, aux


# =============================================================================
# Classical control branch
# =============================================================================
class ClassicalControlBranch(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_qubits: int,
        n_layers: int,
        seed: int,
        projection_mode: str,
        illumination_token_mode: str,
        noise_mode: str,
        noise_scale: float,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.branch_name = "classical_control"
        self.backend_name = "pure_classical_control"
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.use_noise = noise_mode != "none"
        self.use_entangle = True
        self.active_layers = n_layers

        patch_dim = patch_size * patch_size * 3
        embed_dim = 2 * n_qubits

        self.projector = PatchProjector(patch_dim, embed_dim, seed=seed, mode=projection_mode)
        self.illumination = IlluminationEncoder(mode=illumination_token_mode)
        self.noise_mlp = nn.Sequential(
            nn.Linear(embed_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        blocks = []
        for _ in range(n_layers):
            blocks.append(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = F.unfold(x, kernel_size=(p, p), stride=(p, p))
        return patches.transpose(1, 2).contiguous()

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        flat = patches.reshape(b * n, d)
        return torch.tanh(self.projector(flat)).view(b, n, -1)

    def _noise(self, emb: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        if not self.use_noise or self.noise_mode == "none":
            return torch.zeros_like(emb)

        h = torch.cat([emb, tok], dim=1)
        learned = self.noise_mlp(h)

        if self.noise_mode == "sinusoidal":
            return torch.sin(learned) * self.noise_scale
        if self.noise_mode == "linear":
            return learned * self.noise_scale
        if self.noise_mode == "learned":
            return torch.tanh(learned) * self.noise_scale

        raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b = x.shape[0]
        tok = self.illumination(x)
        patches = self.patchify(x)
        emb = self.encode(patches)
        _, n_patches, d = emb.shape

        flat = emb.view(b * n_patches, d)
        tok_rep = tok.unsqueeze(1).expand(b, n_patches, 2).reshape(b * n_patches, 2)

        clean = flat
        noisy = flat + self._noise(flat, tok_rep)
        out = noisy

        for l in range(int(max(0, min(self.n_layers, self.active_layers)))):
            residual = out
            out = self.blocks[l](out)
            if self.use_entangle:
                out = 0.8 * out + 0.2 * torch.roll(out, shifts=1, dims=1)
            out = self.norm(out + residual)

        sim = F.cosine_similarity(clean, out, dim=1).clamp(-1.0, 1.0)
        patch_score = (1.0 - ((sim + 1.0) * 0.5)).view(b, n_patches)
        img_score = patch_score.mean(dim=1)

        aux = {
            "fidelity": ((sim + 1.0) * 0.5).view(b, n_patches),
            "illumination_token": tok,
        }
        return img_score, patch_score, aux


# =============================================================================
# VAE branch
# =============================================================================
class PatchVAE(nn.Module):
    def __init__(self, patch_dim: int, latent_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.dec(z)
        return recon, mu, logvar


def vae_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


# =============================================================================
# Hybrid model
# =============================================================================
class QFidelity(nn.Module):
    def __init__(
        self,
        branch: nn.Module,
        img_size: int,
        patch_size: int,
        vae_latent_dim: int,
        vae_hidden_dim: int,
        fusion_mode: str,
        gen_weight_patch: float,
    ):
        super().__init__()
        self.branch = branch
        self.img_size = img_size
        self.patch_size = patch_size
        self.fusion_mode = fusion_mode
        self.gen_weight_patch = float(gen_weight_patch)

        patch_dim = patch_size * patch_size * 3
        self.vae = PatchVAE(
            patch_dim=patch_dim,
            latent_dim=vae_latent_dim,
            hidden_dim=vae_hidden_dim,
        )

        if fusion_mode == "learned_gate":
            self.fusion_gate = nn.Sequential(
                nn.Linear(2, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
        else:
            self.fusion_gate = None

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = F.unfold(x, kernel_size=(p, p), stride=(p, p))
        return patches.transpose(1, 2).contiguous()

    def vae_patch_error(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self.patchify(x)
        b, n, d = patches.shape
        flat = patches.view(b * n, d)
        recon, mu, logvar = self.vae(flat)
        recon_loss = F.mse_loss(recon, flat, reduction="none").mean(dim=1)
        return recon_loss.view(b, n), mu, logvar

    @staticmethod
    def norm_patch_error(patch_err: torch.Tensor) -> torch.Tensor:
        denom = patch_err.mean(dim=1, keepdim=True) + 1e-6
        scaled = patch_err / denom
        return torch.tanh(scaled).clamp(0.0, 1.0)

    def fuse_patch(self, branch_patch: torch.Tensor, gen_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "fixed":
            w = torch.full_like(branch_patch, self.gen_weight_patch)
        elif self.fusion_mode == "learned_gate":
            g = torch.stack([branch_patch, gen_patch], dim=-1)
            w = torch.sigmoid(self.fusion_gate(g).squeeze(-1))
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        patch_score = (1.0 - w) * branch_patch + w * gen_patch
        return patch_score, w

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, branch_patch, _ = self.branch(x)
        patch_err, _, _ = self.vae_patch_error(x)
        gen_patch = self.norm_patch_error(patch_err)
        patch_score, _ = self.fuse_patch(branch_patch, gen_patch)
        img_score = patch_score.mean(dim=1)
        return img_score, patch_score

    def forward_full(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        branch_img, branch_patch, aux = self.branch(x)
        patch_err, mu, logvar = self.vae_patch_error(x)
        gen_patch = self.norm_patch_error(patch_err)
        patch_score, gate = self.fuse_patch(branch_patch, gen_patch)
        img_score = patch_score.mean(dim=1)

        return {
            "img_score": img_score,
            "patch_score": patch_score,
            "branch_img": branch_img,
            "branch_patch": branch_patch,
            "gen_patch": gen_patch,
            "gate": gate,
            "patch_err": patch_err,
            "mu": mu,
            "logvar": logvar,
            **aux,
        }

    def loss(
        self,
        x: torch.Tensor,
        beta_kl: float,
        gen_lambda: float,
        train_vae_only_on_good: bool,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward_full(x)
        branch_loss = out["branch_patch"].mean()

        if train_vae_only_on_good and labels is not None:
            good_mask = labels == 0
            xg = x[good_mask] if good_mask.any() else x[:0]
        else:
            xg = x

        gen_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        if xg.shape[0] > 0:
            patch_err, mu, logvar = self.vae_patch_error(xg)
            recon = patch_err.mean()
            kl = vae_kl(mu, logvar).mean()
            gen_loss = recon + float(beta_kl) * kl

        total = branch_loss + float(gen_lambda) * gen_loss
        logs = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_branch": float(branch_loss.detach().cpu().item()),
            "loss_gen": float(gen_loss.detach().cpu().item()),
        }
        return total, logs


# =============================================================================
# Model factory
# =============================================================================
def build_model(
    args: argparse.Namespace,
    seed: int,
    override: Optional[Dict[str, object]] = None,
) -> QFidelity:
    cfg = {
        "branch_type": args.branch_type,
        "projection_mode": args.projection_mode,
        "illumination_token_mode": args.illumination_token_mode,
        "noise_mode": args.noise_mode,
        "fusion_mode": args.fusion_mode,
        "noise_scale": args.noise_scale,
    }
    if override is not None:
        cfg.update(override)

    if cfg["branch_type"] == "quantum_simulator":
        branch = QuantumSimulatorBranch(
            patch_size=args.patch_size,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            seed=seed,
            projection_mode=str(cfg["projection_mode"]),
            illumination_token_mode=str(cfg["illumination_token_mode"]),
            noise_mode=str(cfg["noise_mode"]),
            noise_scale=float(cfg["noise_scale"]),
        )
    elif cfg["branch_type"] == "classical_control":
        branch = ClassicalControlBranch(
            patch_size=args.patch_size,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            seed=seed,
            projection_mode=str(cfg["projection_mode"]),
            illumination_token_mode=str(cfg["illumination_token_mode"]),
            noise_mode=str(cfg["noise_mode"]),
            noise_scale=float(cfg["noise_scale"]),
            hidden_dim=args.control_hidden_dim,
        )
    else:
        raise ValueError(f"Unsupported branch_type: {cfg['branch_type']}")

    return QFidelity(
        branch=branch,
        img_size=args.img_size,
        patch_size=args.patch_size,
        vae_latent_dim=args.vae_latent_dim,
        vae_hidden_dim=args.vae_hidden_dim,
        fusion_mode=str(cfg["fusion_mode"]),
        gen_weight_patch=args.gen_weight_patch,
    )


# =============================================================================
# Scoring utilities
# =============================================================================
def patch_to_pixel(patch_scores: torch.Tensor, img_size: int, patch_size: int) -> torch.Tensor:
    b, _ = patch_scores.shape
    g = img_size // patch_size
    pm = patch_scores.view(b, g, g).unsqueeze(1)
    return pm.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)


# =============================================================================
# Training and inference
# =============================================================================
def train_one_category(
    model: QFidelity,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    amp: bool,
    gen_lambda: float,
    beta_kl: float,
    train_vae_only_on_good: bool,
) -> Dict[str, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    best_val = float("inf")
    best_state = None
    epoch_times: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(amp and device.type == "cuda"),
                dtype=autocast_dtype,
            ):
                loss, _ = model.loss(
                    x=x,
                    beta_kl=beta_kl,
                    gen_lambda=gen_lambda,
                    train_vae_only_on_good=train_vae_only_on_good,
                    labels=y,
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = float("nan")

        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            vlosses = []

            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device, non_blocking=True)
                    y = batch["label"].to(device, non_blocking=True)

                    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=(amp and device.type == "cuda"),
                        dtype=autocast_dtype,
                    ):
                        loss, _ = model.loss(
                            x=x,
                            beta_kl=beta_kl,
                            gen_lambda=gen_lambda,
                            train_vae_only_on_good=train_vae_only_on_good,
                            labels=y,
                        )
                    vlosses.append(float(loss.detach().cpu().item()))

            val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        dt = time.time() - t0
        epoch_times.append(dt)
        peak_mem = max_mem_mb(device)

        print(
            f"Epoch {ep}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"time_s={dt:.2f} | "
            f"peak_mem_mb={peak_mem:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": float(best_val) if math.isfinite(best_val) else float("nan"),
        "mean_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else float("nan"),
    }


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    patch_size: int,
    amp: bool,
) -> Dict[str, object]:
    model.eval()

    all_img_scores = []
    all_pix_scores = []
    all_labels = []
    all_masks = []
    all_paths: List[str] = []
    all_defects: List[str] = []
    all_splits: List[str] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].cpu().numpy().astype(np.int32)
            m = batch["mask"].cpu().numpy().astype(np.uint8)

            autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(amp and device.type == "cuda"),
                dtype=autocast_dtype,
            ):
                img_score, patch_score = model(x)
                pix_map = patch_to_pixel(patch_score, img_size, patch_size)

            all_img_scores.append(img_score.detach().float().cpu().numpy())
            all_pix_scores.append(pix_map.detach().float().cpu().numpy())
            all_labels.append(y)
            all_masks.append(m)
            all_paths.extend(batch["path"])
            all_defects.extend(batch["defect"])
            all_splits.extend(batch["split"])

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()

    img_scores = np.concatenate(all_img_scores, axis=0) if all_img_scores else np.array([], dtype=np.float32)
    pix_scores = np.concatenate(all_pix_scores, axis=0) if all_pix_scores else np.array([], dtype=np.float32)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int32)
    masks = np.concatenate(all_masks, axis=0) if all_masks else np.array([], dtype=np.uint8)

    total_images = int(img_scores.shape[0])
    total_time_s = float(t1 - t0)
    total_time_ms = float(total_time_s * 1000.0)
    fps = float(total_images / total_time_s) if total_time_s > 0 else float("nan")

    return {
        "img_scores": img_scores,
        "pix_scores": pix_scores,
        "labels": labels,
        "masks": masks,
        "paths": all_paths,
        "defects": all_defects,
        "splits": all_splits,
        "inference_time_ms": total_time_ms,
        "fps": fps,
        "n_images": total_images,
        "peak_memory_mb": max_mem_mb(device),
    }


# =============================================================================
# Figures
# =============================================================================
FIG_BRANCH_PDF = "quantum_circuit.pdf"
FIG_QUAL_PDF = "qualitative_results.pdf"

FIG1_TITLE_SIZE = 12
FIG1_LABEL_SIZE = 11
FIG1_TICK_SIZE = 10
FIG1_ROW_TITLE_SIZE = 12
BRANCH_TITLE_SIZE = 20
BRANCH_BOX_TEXT_SIZE = 11
BRANCH_NODE_TEXT_SIZE = 12


def _read_fig_img(path: str, img_size: int) -> np.ndarray:
    x = read_image(path)
    if x.ndim != 3:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x.float().div(255.0).unsqueeze(0)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.squeeze(0).clamp(0.0, 1.0)
    return np.transpose(x.cpu().numpy(), (1, 2, 0))


def _norm_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = hm.astype(np.float32)
    mn = float(hm.min())
    mx = float(hm.max())
    if mx - mn < 1e-12:
        return np.zeros_like(hm, dtype=np.float32)
    return (hm - mn) / (mx - mn)


def pick_examples(labels: np.ndarray, scores: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    if scores.size == 0:
        return None, None

    good_idx = np.where(labels == 0)[0]
    anom_idx = np.where(labels == 1)[0]

    ig = int(good_idx[np.argmin(scores[good_idx])]) if good_idx.size > 0 else int(np.argmin(scores))
    ia = int(anom_idx[np.argmax(scores[anom_idx])]) if anom_idx.size > 0 else int(np.argmax(scores))
    return ig, ia


def save_quantum_circuit(out_path_pdf: str, branch_type: str, n_qubits: int, n_layers: int) -> None:
    fig = plt.figure(figsize=(14, max(4, 0.6 * n_qubits)))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    if branch_type == "quantum_simulator":
        y0 = np.arange(n_qubits)[::-1].astype(float)

        for i in range(n_qubits):
            ax.plot([0, 10 + 2 * n_layers], [y0[i], y0[i]], linewidth=1.5)

        def box(x: float, y: float, txt: str) -> None:
            ax.add_patch(plt.Rectangle((x - 0.35, y - 0.25), 0.7, 0.5, fill=False, linewidth=1.5))
            ax.text(x, y, txt, ha="center", va="center", fontsize=BRANCH_BOX_TEXT_SIZE)

        x = 0.8
        for q in range(n_qubits):
            box(x, y0[q], "RY")
            box(x + 0.8, y0[q], "RZ")

        x = 2.4
        for q in range(n_qubits):
            box(x, y0[q], "Noise")

        x = 4.0
        for _ in range(n_layers):
            for q in range(n_qubits):
                box(x, y0[q], "RY")
                box(x + 0.8, y0[q], "RZ")

            cx = x + 1.6
            for q in range(n_qubits - 1):
                ax.plot([cx, cx], [y0[q], y0[q + 1]], linewidth=1.5)
                ax.plot([cx - 0.1, cx + 0.1], [y0[q], y0[q]], linewidth=1.5)
                ax.add_patch(plt.Circle((cx, y0[q + 1]), 0.06, fill=True))

            ax.plot([cx, cx], [y0[n_qubits - 1], y0[0]], linewidth=1.5)
            ax.plot([cx - 0.1, cx + 0.1], [y0[n_qubits - 1], y0[n_qubits - 1]], linewidth=1.5)
            ax.add_patch(plt.Circle((cx, y0[0]), 0.06, fill=True))
            x += 2.0

        ax.set_xlim(0, 10 + 2 * n_layers)
        ax.set_ylim(-1, n_qubits)

    else:
        ax.text(
            0.5,
            0.86,
            "Classical Control Branch",
            ha="center",
            va="center",
            fontsize=BRANCH_TITLE_SIZE,
            fontweight="bold",
        )

        texts = [
            (0.15, 0.60, "Patch Projector"),
            (0.38, 0.60, "Illumination\nEncoder"),
            (0.62, 0.60, "Residual Control\nBlocks"),
            (0.85, 0.60, "Similarity\nScore"),
        ]

        for x, y, t in texts:
            ax.add_patch(plt.Rectangle((x - 0.09, y - 0.09), 0.18, 0.18, fill=False, linewidth=2))
            ax.text(x, y, t, ha="center", va="center", fontsize=BRANCH_NODE_TEXT_SIZE)

        for x1, x2 in [(0.24, 0.29), (0.47, 0.53), (0.71, 0.76)]:
            ax.annotate("", xy=(x2, 0.60), xytext=(x1, 0.60), arrowprops=dict(arrowstyle="->", lw=2))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path_pdf, dpi=300)
    plt.close(fig)


def save_figure_1(results_dir: str, categories: List[str], img_size: int) -> None:
    rows = [r for r in _GLOBAL_QUAL_ROWS if r.get("category") in set(categories)]
    rows.sort(key=lambda d: str(d.get("category", "")))

    if not rows:
        return

    nrows = len(rows)
    fig = plt.figure(figsize=(12.6, max(4.6, 1.55 * nrows)), constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=5, wspace=0.02, hspace=0.15)

    for r, item in enumerate(rows):
        category = str(item["category"])
        good_path = str(item["good_path"])
        anom_path = str(item["anom_path"])
        good_score = float(item["good_score"])
        anom_score = float(item["anom_score"])

        good_hm = _norm_heatmap(np.asarray(item["good_hm"], dtype=np.float32))
        anom_hm = _norm_heatmap(np.asarray(item["anom_hm"], dtype=np.float32))
        good_scores = np.asarray(item["scores_good"], dtype=np.float32)
        anom_scores = np.asarray(item["scores_anom"], dtype=np.float32)

        img_good = _read_fig_img(good_path, img_size)
        img_anom = _read_fig_img(anom_path, img_size)

        ax1 = fig.add_subplot(gs[r, 0])
        ax1.imshow(img_good)
        ax1.axis("off")
        ax1.set_title(f"Good | score={good_score:.4f}", fontsize=FIG1_TITLE_SIZE, pad=4)

        ax2 = fig.add_subplot(gs[r, 1])
        ax2.imshow(img_good)
        ax2.imshow(good_hm, alpha=0.55, cmap="jet")
        ax2.axis("off")
        ax2.set_title("Good + heatmap", fontsize=FIG1_TITLE_SIZE, pad=4)

        ax3 = fig.add_subplot(gs[r, 2])
        ax3.imshow(img_anom)
        ax3.axis("off")
        ax3.set_title(f"Anomalous | score={anom_score:.4f}", fontsize=FIG1_TITLE_SIZE, pad=4)

        ax4 = fig.add_subplot(gs[r, 3])
        ax4.imshow(img_anom)
        ax4.imshow(anom_hm, alpha=0.55, cmap="jet")
        ax4.axis("off")
        ax4.set_title("Anomalous + heatmap", fontsize=FIG1_TITLE_SIZE, pad=4)

        ax5 = fig.add_subplot(gs[r, 4])
        data, pos, labels = [], [], []

        if good_scores.size > 0:
            data.append(good_scores)
            pos.append(1)
            labels.append("Good")

        if anom_scores.size > 0:
            data.append(anom_scores)
            pos.append(2)
            labels.append("Anomalous")

        if data:
            bp = ax5.boxplot(
                data,
                vert=False,
                positions=pos,
                widths=0.6,
                patch_artist=False,
                showfliers=False,
                whis=(5, 95),
            )
            for med in bp.get("medians", []):
                med.set_linewidth(1.6)

        ax5.set_yticks(pos if pos else [])
        ax5.set_yticklabels(labels if labels else [], fontsize=FIG1_TICK_SIZE)
        ax5.set_xlabel("Image score", fontsize=FIG1_LABEL_SIZE)
        ax5.tick_params(axis="x", labelsize=FIG1_TICK_SIZE)
        ax5.tick_params(axis="y", labelsize=FIG1_TICK_SIZE)
        ax5.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax5.set_title(category, fontsize=FIG1_ROW_TITLE_SIZE, pad=4)

    fig.savefig(os.path.join(results_dir, FIG_QUAL_PDF), dpi=300)
    plt.close(fig)


# =============================================================================
# Outputs
# =============================================================================
def save_category_outputs(
    paths: Dict[str, str],
    category: str,
    summary_rows: List[Dict[str, object]],
    pred_rows: List[Dict[str, object]],
    speed_rows: List[Dict[str, object]],
    ablation_rows: List[Dict[str, object]],
) -> None:
    df_sum = pd.DataFrame(summary_rows)
    if len(df_sum) > 0:
        df_sum.insert(0, "category", category)
    df_sum.to_csv(paths["csv_summary"], index=False)

    pd.DataFrame(pred_rows).to_csv(paths["csv_predictions"], index=False)
    pd.DataFrame(speed_rows).to_csv(paths["csv_speed"], index=False)
    pd.DataFrame(ablation_rows).to_csv(paths["csv_ablation"], index=False)


# =============================================================================
# Ablation
# =============================================================================
def make_ablation_presets(args: argparse.Namespace) -> List[Tuple[str, Dict[str, object]]]:
    return [
        ("full", {}),
        ("classicalcontrol", {"branch_type": "classical_control"}),
        ("trainableprojection", {"projection_mode": "trainable_linear"}),
        ("learnedfusion", {"fusion_mode": "learned_gate"}),
        ("nonoise", {"noise_mode": "none"}),
        ("notoken", {"illumination_token_mode": "none"}),
    ]


def eval_variant(
    name: str,
    model: QFidelity,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    patch_size: int,
    amp: bool,
) -> Dict[str, object]:
    out = run_inference(model, loader, device, img_size, patch_size, amp)
    metrics = eval_public_metrics(out["labels"], out["img_scores"], out["masks"], out["pix_scores"])

    return {
        "variant": name,
        "ImageAUROC": metrics["ImageAUROC"],
        "PixelAUROC": metrics["PixelAUROC"],
        "ClassF1_public_pct": metrics["ClassF1_public_pct"],
        "fps": out["fps"],
        "inference_time_ms": out["inference_time_ms"],
        "peak_memory_mb": out["peak_memory_mb"],
    }


def pretty_variant(name: str) -> str:
    mapping = {
        "full": "Full",
        "classicalcontrol": "Classical Control",
        "trainableprojection": "Trainable Projection",
        "learnedfusion": "Learned Fusion",
        "nonoise": "No Noise",
        "notoken": "No Token",
    }
    return mapping.get(name, name)


# =============================================================================
# Category pipeline
# =============================================================================
def run_category_pipeline(args: argparse.Namespace, category: str, seed: int) -> Dict[str, float]:
    paths = make_paths(args.results_dir, category, seed)
    index = index_mvtec_ad2(args.data_root, category)
    log_data_stats(index, category)

    preprocess_cats = parse_csv_set(args.preprocess_categories)

    train_ds = MVTecAD2Dataset(index.get("train", []), args.img_size, category, preprocess_cats)
    val_items = index.get("validation", [])
    pub_items = index.get("test_public", [])

    val_ds = MVTecAD2Dataset(val_items, args.img_size, category, preprocess_cats) if len(val_items) > 0 else None
    pub_ds = MVTecAD2Dataset(pub_items, args.img_size, category, preprocess_cats) if len(pub_items) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
        drop_last=False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_batch,
            drop_last=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = build_model(args, seed).to(device)

    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    params_m = count_params_m(model)
    first_batch = next(iter(train_loader))

    print(
        f"\nInput batch | category={category} | "
        f"image_shape={tuple(first_batch['image'].shape)} | "
        f"mask_shape={tuple(first_batch['mask'].shape)}"
    )
    print(
        f"Run started | category={category} | seed={seed} | device={device.type} | "
        f"params_m={params_m:.6f} | branch={model.branch.branch_name} | "
        f"backend={model.branch.backend_name}"
    )

    _ = train_one_category(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        amp=args.amp,
        gen_lambda=args.gen_lambda,
        beta_kl=args.beta_kl,
        train_vae_only_on_good=args.train_vae_only_on_good,
    )

    summary_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    speed_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []

    if pub_ds is not None and len(pub_ds) > 0:
        pub_loader = DataLoader(
            pub_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_batch,
            drop_last=False,
        )

        print(f"\nEvaluating split | category={category} | seed={seed} | split=test_public")
        out = run_inference(model, pub_loader, device, args.img_size, args.patch_size, args.amp)
        metrics = eval_public_metrics(out["labels"], out["img_scores"], out["masks"], out["pix_scores"])

        print(
            f"Public metrics | split=test_public | "
            f"ImageAUROC={metrics['ImageAUROC']:.6f} | "
            f"PixelAUROC={metrics['PixelAUROC']:.6f} | "
            f"ClassF1_pct={metrics['ClassF1_public_pct']:.2f}"
        )

        summary_rows.append(
            {
                "seed": seed,
                "split": "test_public",
                "ImageAUROC": metrics["ImageAUROC"],
                "PixelAUROC": metrics["PixelAUROC"],
                "ClassF1_public_pct": metrics["ClassF1_public_pct"],
                "fps": out["fps"],
                "inference_time_ms": out["inference_time_ms"],
                "peak_memory_mb": out["peak_memory_mb"],
            }
        )

        speed_rows.append(
            {
                "category": category,
                "seed": seed,
                "split": "test_public",
                "n_images": out["n_images"],
                "fps": out["fps"],
                "inference_time_ms": out["inference_time_ms"],
                "peak_memory_mb": out["peak_memory_mb"],
            }
        )

        for i in range(len(out["paths"])):
            pred_rows.append(
                {
                    "category": category,
                    "seed": seed,
                    "split": "test_public",
                    "path": out["paths"][i],
                    "defect": out["defects"][i],
                    "label": int(out["labels"][i]) if out["labels"].size > 0 else -1,
                    "image_score": float(out["img_scores"][i]) if out["img_scores"].size > 0 else float("nan"),
                }
            )

        labels = out["labels"].astype(np.int32)
        scores = out["img_scores"].astype(np.float32)
        pix = out["pix_scores"].astype(np.float32)
        paths_ = out["paths"]

        good_scores = scores[labels == 0]
        anom_scores = scores[labels == 1]
        ig, ia = pick_examples(labels, scores)

        if ig is not None and ia is not None and seed == parse_csv_ints(args.seeds)[0]:
            _GLOBAL_QUAL_ROWS.append(
                {
                    "category": category,
                    "good_path": paths_[ig],
                    "anom_path": paths_[ia],
                    "good_score": float(scores[ig]),
                    "anom_score": float(scores[ia]),
                    "good_hm": pix[ig, 0],
                    "anom_hm": pix[ia, 0],
                    "scores_good": good_scores.copy(),
                    "scores_anom": anom_scores.copy(),
                }
            )

        if args.retrain_ablation:
            presets = make_ablation_presets(args)
            for variant_name, override in presets:
                if variant_name == "full":
                    ablation_rows.append(
                        eval_variant(
                            variant_name,
                            model,
                            pub_loader,
                            device,
                            args.img_size,
                            args.patch_size,
                            args.amp,
                        )
                    )
                    continue

                v_model = build_model(args, seed, override=override).to(device)
                _ = train_one_category(
                    model=v_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=args.ablation_epochs,
                    lr=args.lr,
                    amp=args.amp,
                    gen_lambda=args.gen_lambda,
                    beta_kl=args.beta_kl,
                    train_vae_only_on_good=args.train_vae_only_on_good,
                )
                ablation_rows.append(
                    eval_variant(
                        variant_name,
                        v_model,
                        pub_loader,
                        device,
                        args.img_size,
                        args.patch_size,
                        args.amp,
                    )
                )
                del v_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    save_category_outputs(paths, category, summary_rows, pred_rows, speed_rows, ablation_rows)

    out_summary = {
        "category": category,
        "seed": seed,
        "ImageAUROC": float("nan"),
        "PixelAUROC": float("nan"),
        "ClassF1_public_pct": float("nan"),
        "fps": float("nan"),
        "inference_time_ms": float("nan"),
        "peak_memory_mb": float("nan"),
    }

    if len(summary_rows) > 0:
        row = summary_rows[0]
        out_summary.update(
            {
                "ImageAUROC": float(row["ImageAUROC"]),
                "PixelAUROC": float(row["PixelAUROC"]),
                "ClassF1_public_pct": float(row["ClassF1_public_pct"]),
                "fps": float(row["fps"]),
                "inference_time_ms": float(row["inference_time_ms"]),
                "peak_memory_mb": float(row["peak_memory_mb"]),
            }
        )

    return out_summary


# =============================================================================
# Seed-level CSV tables
# =============================================================================
def save_seed_results(results_dir: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "seed_level_results.csv"), index=False)

    metric_cols = [
        "ImageAUROC",
        "PixelAUROC",
        "ClassF1_public_pct",
        "fps",
        "inference_time_ms",
        "peak_memory_mb",
    ]

    per_seed_rows = []
    for seed, grp in df.groupby("seed"):
        row = {"seed": int(seed)}
        for col in metric_cols:
            row[col] = float(np.nanmean(grp[col].to_numpy(dtype=np.float64)))
        per_seed_rows.append(row)

    pd.DataFrame(per_seed_rows).to_csv(os.path.join(results_dir, "seed_mean_results.csv"), index=False)

    per_cat_rows = []
    for (category, seed), grp in df.groupby(["category", "seed"]):
        row = {"category": category, "seed": int(seed)}
        for col in metric_cols:
            row[col] = float(np.nanmean(grp[col].to_numpy(dtype=np.float64)))
        per_cat_rows.append(row)

    pd.DataFrame(per_cat_rows).to_csv(os.path.join(results_dir, "seed_category_results.csv"), index=False)


def save_seed_ablation_tables(results_dir: str, categories: List[str], seeds: List[int]) -> None:
    variant_order = [
        "full",
        "classicalcontrol",
        "trainableprojection",
        "learnedfusion",
        "nonoise",
        "notoken",
    ]

    for seed in seeds:
        rows = []

        for category in categories:
            path = os.path.join(
                results_dir,
                category,
                f"seed_{seed}",
                f"{category}_seed_{seed}_ablation.csv",
            )
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            if len(df) == 0:
                continue

            cols = [
                "variant",
                "ImageAUROC",
                "PixelAUROC",
                "ClassF1_public_pct",
                "fps",
                "inference_time_ms",
                "peak_memory_mb",
            ]
            keep = [c for c in cols if c in df.columns]
            rows.append(df[keep].copy())

        if not rows:
            continue

        full_df = pd.concat(rows, axis=0, ignore_index=True)
        out_rows = []

        for variant in variant_order:
            part = full_df[full_df["variant"] == variant]
            if len(part) == 0:
                continue

            out_rows.append(
                {
                    "Variant": pretty_variant(variant),
                    "Image AUROC": float(np.nanmean(part["ImageAUROC"].to_numpy(dtype=np.float64))),
                    "Pixel AUROC": float(np.nanmean(part["PixelAUROC"].to_numpy(dtype=np.float64))),
                    "Class F1 (%)": float(np.nanmean(part["ClassF1_public_pct"].to_numpy(dtype=np.float64))),
                    "FPS": float(np.nanmean(part["fps"].to_numpy(dtype=np.float64))),
                    "Inference Time (ms)": float(np.nanmean(part["inference_time_ms"].to_numpy(dtype=np.float64))),
                    "Peak Memory (MB)": float(np.nanmean(part["peak_memory_mb"].to_numpy(dtype=np.float64))),
                }
            )

        out_df = pd.DataFrame(out_rows)

        if len(out_df) == 0:
            continue

        out_df["Image AUROC"] = out_df["Image AUROC"].round(4)
        out_df["Pixel AUROC"] = out_df["Pixel AUROC"].round(4)
        out_df["Class F1 (%)"] = out_df["Class F1 (%)"].round(2)
        out_df["FPS"] = out_df["FPS"].round(2)
        out_df["Inference Time (ms)"] = out_df["Inference Time (ms)"].round(2)
        out_df["Peak Memory (MB)"] = out_df["Peak Memory (MB)"].round(2)

        out_path = os.path.join(results_dir, f"ablation_aggregate_seed_{seed}.csv")
        out_df.to_csv(out_path, index=False)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid anomaly detection pipeline with simulated quantum and classical control branches."
    )

    p.add_argument("--data_root", type=str, default="Dataset")
    p.add_argument("--results_dir", type=str, default="Results")
    p.add_argument(
        "--categories",
        type=str,
        default="can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts",
    )
    p.add_argument(
        "--preprocess_categories",
        type=str,
        default="can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts",
    )

    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--n_qubits", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--noise_scale", type=float, default=0.35)

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--ablation_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seeds", type=str, default="7")

    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--retrain_ablation", action="store_true")

    p.add_argument(
        "--branch_type",
        type=str,
        default="quantum_simulator",
        choices=["quantum_simulator", "classical_control"],
    )
    p.add_argument(
        "--projection_mode",
        type=str,
        default="frozen_random",
        choices=["frozen_random", "trainable_linear"],
    )
    p.add_argument(
        "--illumination_token_mode",
        type=str,
        default="stats",
        choices=["none", "stats", "learned"],
    )
    p.add_argument(
        "--noise_mode",
        type=str,
        default="sinusoidal",
        choices=["none", "sinusoidal", "linear", "learned"],
    )
    p.add_argument(
        "--fusion_mode",
        type=str,
        default="fixed",
        choices=["fixed", "learned_gate"],
    )

    p.add_argument("--vae_latent_dim", type=int, default=64)
    p.add_argument("--vae_hidden_dim", type=int, default=512)
    p.add_argument("--control_hidden_dim", type=int, default=64)
    p.add_argument("--gen_weight_patch", type=float, default=0.35)
    p.add_argument("--gen_lambda", type=float, default=0.25)
    p.add_argument("--beta_kl", type=float, default=1e-3)
    p.add_argument("--train_vae_only_on_good", action="store_true")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = parse_csv_ints(args.seeds)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Output directory: {args.results_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    branch_pdf = os.path.join(args.results_dir, FIG_BRANCH_PDF)
    save_quantum_circuit(branch_pdf, args.branch_type, args.n_qubits, args.n_layers)

    run_rows: List[Dict[str, float]] = []

    for seed in seeds:
        set_seed(seed)
        for category in categories:
            result = run_category_pipeline(args, category, seed)
            run_rows.append(result)

    save_seed_results(args.results_dir, run_rows)
    save_seed_ablation_tables(args.results_dir, categories, seeds)
    save_figure_1(args.results_dir, categories, args.img_size)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()