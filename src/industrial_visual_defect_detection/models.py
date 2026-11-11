from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class Teacher(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            x = self.stem(x)
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
        return a, b, c


class DSBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.GroupNorm(min(8, cin), cin), nn.SiLU(),
            nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Student(nn.Module):
    """Small multi-scale student with projections to ResNet18 feature widths."""
    def __init__(self, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, width), nn.SiLU())
        self.s1 = DSBlock(width, width * 2, 2)
        self.s2 = DSBlock(width * 2, width * 3, 2)
        self.s3 = DSBlock(width * 3, width * 4, 2)
        self.p1 = nn.Conv2d(width * 2, 64, 1)
        self.p2 = nn.Conv2d(width * 3, 128, 1)
        self.p3 = nn.Conv2d(width * 4, 256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        h1 = self.s1(x)
        h2 = self.s2(h1)
        h3 = self.s3(h2)
        return self.p1(h1), self.p2(h2), self.p3(h3)


def distillation_loss(teacher: tuple[torch.Tensor, ...], student: tuple[torch.Tensor, ...], weights: tuple[float, ...] = (1., 1., 1.)) -> tuple[torch.Tensor, list[torch.Tensor]]:
    losses = []
    for target, pred in zip(teacher, student):
        target = F.normalize(target, dim=1)
        pred = F.normalize(pred, dim=1)
        losses.append((target - pred).square().mean())
    total = torch.stack([w * loss for w, loss in zip(weights, losses)]).sum() / sum(weights)
    return total, losses


def feature_map_scales(teacher: tuple[torch.Tensor, ...], student: tuple[torch.Tensor, ...], output_size: tuple[int, int]) -> tuple[torch.Tensor, ...]:
    maps = []
    for t, s in zip(teacher, student):
        d = (F.normalize(t, dim=1) - F.normalize(s, dim=1)).square().mean(dim=1, keepdim=True)
        maps.append(F.interpolate(d, size=output_size, mode="bilinear", align_corners=False))
    return tuple(maps)


def feature_maps(teacher: tuple[torch.Tensor, ...], student: tuple[torch.Tensor, ...], output_size: tuple[int, int], scale_stats: list[dict[str, float]] | None = None) -> torch.Tensor:
    maps = feature_map_scales(teacher, student, output_size)
    if scale_stats:
        maps = tuple(((m - float(stats["mean"])) / max(float(stats["std"]), 1e-8)).clamp_min(0.) for m, stats in zip(maps, scale_stats))
    return torch.stack(maps, dim=0).mean(dim=0)


@torch.no_grad()
def chunked_min_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Return exact nearest-reference Euclidean distances with bounded query chunks."""
    if queries.ndim != 3 or bank.ndim != 2 or queries.shape[-1] != bank.shape[-1]:
        raise ValueError("queries must be [batch, patches, dimensions] and bank [references, dimensions]")
    if bank.shape[0] == 0 or chunk <= 0:
        raise ValueError("bank must be non-empty and chunk must be positive")
    device = queries.device
    search_device = torch.device("cpu") if device.type == "mps" else device
    reference = bank.to(search_device)
    distances = []
    for start in range(0, queries.shape[1], chunk):
        query = queries[:, start:start + chunk].to(search_device)
        nearest = torch.cdist(query, reference).min(dim=-1).values
        distances.append(nearest.to(device))
    return torch.cat(distances, dim=1)


class PatchMemory(nn.Module):
    """ResNet18 multi-scale feature bank with seeded reservoir-style random coreset."""
    def __init__(self, max_points: int = 10000):
        super().__init__()
        self.backbone = Teacher(pretrained=True)
        self.max_points = max_points
        self.register_buffer("bank", torch.empty(0, 448))

    @torch.no_grad()
    def fit(self, batches, device: torch.device, seed: int = 42) -> None:
        candidates = []
        batch_count = max(1, len(batches))
        per_batch = max(1, (self.max_points + batch_count - 1) // batch_count)
        generator = torch.Generator().manual_seed(seed)
        for batch in batches:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            fs = self.backbone(x.to(device))
            aligned = [F.adaptive_avg_pool2d(v, fs[1].shape[-2:]) for v in fs]
            z = torch.cat(aligned, dim=1).permute(0, 2, 3, 1).reshape(-1, 448).cpu()
            chosen = torch.randperm(len(z), generator=generator)[:per_batch]
            candidates.append(z[chosen])
        all_features = torch.cat(candidates)
        if len(all_features) > self.max_points:
            all_features = all_features[torch.randperm(len(all_features), generator=generator)[:self.max_points]]
        self.bank = all_features.to(device)

    @torch.no_grad()
    def predict_map(self, x: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        fs = self.backbone(x)
        aligned = [F.adaptive_avg_pool2d(v, fs[1].shape[-2:]) for v in fs]
        q = torch.cat(aligned, dim=1).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 448)
        # torch.cdist support varies by MPS/PyTorch release; keep only this operation on CPU there.
        d = chunked_min_distance(q, self.bank, chunk).reshape(x.shape[0], 1, *fs[1].shape[-2:])
        return F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
