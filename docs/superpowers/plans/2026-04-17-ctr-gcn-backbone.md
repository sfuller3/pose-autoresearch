# CTR-GCN Skeleton Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CTR-GCN-style two-body graph backbone (`STGCNClassifier`) selectable via `POSE_BACKBONE=gcn`, trained side-by-side against the existing temporal CNN.

**Architecture:** A unified 34-joint graph (primary + neighbor skeletons with 6 cross-body seed edges) processed by 4 stages of channel-wise topology refinement (CTRGraphBlock) alternating with multi-scale temporal convolution (GraphTemporalBlock). Input contract `(B, T, 105)` is unchanged, so datasets and the streaming pipeline are untouched. Spec: `docs/superpowers/specs/2026-04-17-ctr-gcn-backbone-design.md`.

**Tech Stack:** PyTorch (pure, no new deps). Tests via `.venv/bin/python -m pytest` (Python 3.9 — no 3.10+ syntax in new code).

**Conventions:**
- All commands run from the repo root.
- `prepare.py` is IMMUTABLE — never modify it.
- Existing 70 tests in `tests/test_pipeline.py` must keep passing after every task.

---

### Task 1: Two-body adjacency in graph.py

**Files:**
- Modify: `pose_autoresearch/graph.py` (append at end)
- Test: `tests/test_pipeline.py` (append new class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# ============================================================================
# CTR-GCN BACKBONE TESTS
# ============================================================================


class TestTwoBodyGraph:
    """Tests for the 34-joint two-person adjacency (CTR-GCN backbone)."""

    def test_shape_and_dtype(self):
        from pose_autoresearch.graph import get_two_body_adjacency
        A = get_two_body_adjacency()
        assert A.shape == (34, 34)
        assert A.dtype == np.float32

    def test_symmetric(self):
        from pose_autoresearch.graph import get_two_body_adjacency
        A = get_two_body_adjacency()
        assert np.allclose(A, A.T, atol=1e-6)

    def test_intra_body_edges_present(self):
        from pose_autoresearch.graph import get_two_body_adjacency
        A = get_two_body_adjacency()
        # left_shoulder <-> right_shoulder for both bodies
        assert A[5, 6] > 0
        assert A[5 + 17, 6 + 17] > 0

    def test_cross_body_edges_present(self):
        from pose_autoresearch.graph import get_two_body_adjacency, TWO_BODY_CROSS_EDGES
        A = get_two_body_adjacency()
        for i, j in TWO_BODY_CROSS_EDGES:
            assert A[i, j] > 0, f"cross edge ({i},{j}) missing"

    def test_no_spurious_cross_edges(self):
        from pose_autoresearch.graph import get_two_body_adjacency
        A = get_two_body_adjacency()
        # primary ankle (15) and neighbor ankle (32) are not connected
        assert A[15, 32] == 0

    def test_normalized(self):
        from pose_autoresearch.graph import get_two_body_adjacency
        A = get_two_body_adjacency()
        # Symmetric normalization keeps values in (0, 1]
        assert A.max() <= 1.0 + 1e-6
        assert A.min() >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestTwoBodyGraph -v`
Expected: FAIL with `ImportError: cannot import name 'get_two_body_adjacency'`

- [ ] **Step 3: Implement in graph.py**

Append to `pose_autoresearch/graph.py`:

```python
# Cross-body seed edges for the unified two-person graph.
# Primary joints are 0-16, neighbor joints are 17-33 (COCO index + 17).
TWO_BODY_CROSS_EDGES = [
    (11, 11 + NUM_JOINTS), (12, 12 + NUM_JOINTS),   # hip <-> hip
    (9, 9 + NUM_JOINTS), (10, 10 + NUM_JOINTS),     # wrist <-> wrist
    (9, 0 + NUM_JOINTS), (10, 0 + NUM_JOINTS),      # primary wrists <-> neighbor nose
]


def get_two_body_adjacency(num_joints: int = NUM_JOINTS) -> np.ndarray:
    """Symmetric-normalized adjacency for a unified two-person skeleton graph.

    Joints 0..16 are the primary person, 17..33 the neighbor. Intra-body
    edges duplicate COCO_17_EDGES at a +17 offset; cross-body seed edges
    connect hips, wrists, and primary-wrist-to-neighbor-head so interaction
    geometry (aggression, working together) is reachable in one hop. The
    CTR mechanism refines this base dynamically at runtime.

    Returns:
        (34, 34) float32, D^{-1/2} (A + I) D^{-1/2} normalized.
    """
    edges = list(COCO_17_EDGES)
    edges += [(i + num_joints, j + num_joints) for i, j in COCO_17_EDGES]
    edges += TWO_BODY_CROSS_EDGES
    A = get_adjacency_matrix(edges, num_joints * 2, self_loops=True)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-6)))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestTwoBodyGraph -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add pose_autoresearch/graph.py tests/test_pipeline.py
git commit -m "feat: two-body 34-joint adjacency with cross-person seed edges"
```

---

### Task 2: GraphTemporalBlock + SqueezeExcitation2d in train.py

**Files:**
- Modify: `train.py` (add after the existing `SqueezeExcitation` class)
- Test: `tests/test_pipeline.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestGraphTemporalBlock:
    """Multi-scale temporal conv over (B, C, T, V) graph features."""

    def test_output_shape_stride1(self):
        from train import GraphTemporalBlock
        block = GraphTemporalBlock(64, stride=1)
        block.eval()
        x = torch.randn(2, 64, 150, 34)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 64, 150, 34)

    def test_output_shape_stride2(self):
        from train import GraphTemporalBlock
        block = GraphTemporalBlock(64, stride=2)
        block.eval()
        x = torch.randn(2, 64, 150, 34)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 64, 75, 34)

    def test_stride2_chains(self):
        """150 -> 75 -> 38 like the spec's stage layout."""
        from train import GraphTemporalBlock
        b1 = GraphTemporalBlock(32, stride=2)
        b2 = GraphTemporalBlock(32, stride=2)
        b1.eval(); b2.eval()
        x = torch.randn(1, 32, 150, 34)
        with torch.no_grad():
            out = b2(b1(x))
        assert out.shape == (1, 32, 38, 34)

    def test_gradient_flow(self):
        from train import GraphTemporalBlock
        block = GraphTemporalBlock(32)
        x = torch.randn(2, 32, 50, 34, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestGraphTemporalBlock -v`
Expected: FAIL with `ImportError: cannot import name 'GraphTemporalBlock'`

- [ ] **Step 3: Implement in train.py**

Insert after the existing `SqueezeExcitation` class in `train.py`:

```python
class SqueezeExcitation2d(nn.Module):
    """Channel attention for (B, C, T, V) graph feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T, V)
        w = self.se(x).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return x * w


class GraphTemporalBlock(nn.Module):
    """Multi-scale temporal convolution over graph features (B, C, T, V).

    Same kernels-3/7/15 + SE design as MultiScaleTemporalBlock, but kernels
    run along T only ((k, 1) Conv2d) so every joint keeps its own temporal
    stream. Channel count is preserved; stride downsamples T.
    """

    def __init__(self, channels, kernels=(3, 7, 15), stride=1, dropout=0.3):
        super().__init__()
        branch_ch = channels // len(kernels)
        remainder = channels - branch_ch * len(kernels)

        self.branches = nn.ModuleList()
        for i, k in enumerate(kernels):
            ch = branch_ch + (remainder if i == 0 else 0)
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, ch, (k, 1), stride=(stride, 1),
                          padding=((k - 1) // 2, 0)),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            ))

        self.conv2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcitation2d(channels)

        if stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )
        else:
            self.residual = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return self.act(x + res)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestGraphTemporalBlock -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_pipeline.py
git commit -m "feat: GraphTemporalBlock — multi-scale temporal conv for graph features"
```

---

### Task 3: CTRGraphBlock in train.py

**Files:**
- Modify: `train.py` (add after `GraphTemporalBlock`)
- Test: `tests/test_pipeline.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestCTRGraphBlock:
    """Channel-wise topology refinement spatial unit."""

    def _make_block(self, in_ch=6, out_ch=64):
        from train import CTRGraphBlock
        from pose_autoresearch.graph import get_two_body_adjacency
        return CTRGraphBlock(in_ch, out_ch, get_two_body_adjacency())

    def test_output_shape(self):
        block = self._make_block()
        block.eval()
        x = torch.randn(2, 6, 150, 34)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 64, 150, 34)

    def test_alpha_initialized_to_zero(self):
        """Identity-start: dynamic refinement begins disabled."""
        block = self._make_block()
        assert torch.all(block.alpha == 0)

    def test_identity_start_ignores_affinity_weights(self):
        """With alpha=0, perturbing theta/phi must not change the output."""
        block = self._make_block()
        block.eval()
        x = torch.randn(2, 6, 50, 34)
        with torch.no_grad():
            out_a = block(x)
            block.theta.weight.add_(1.0)
            block.phi.weight.add_(-1.0)
            out_b = block(x)
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_alpha_changes_output_when_nonzero(self):
        block = self._make_block()
        block.eval()
        x = torch.randn(2, 6, 50, 34)
        with torch.no_grad():
            out_a = block(x)
            block.alpha.add_(0.5)
            out_b = block(x)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_gradient_reaches_alpha(self):
        block = self._make_block()
        x = torch.randn(2, 6, 50, 34)
        block(x).sum().backward()
        assert block.alpha.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestCTRGraphBlock -v`
Expected: FAIL with `ImportError: cannot import name 'CTRGraphBlock'`

- [ ] **Step 3: Implement in train.py**

Insert after `GraphTemporalBlock` in `train.py`:

```python
class CTRGraphBlock(nn.Module):
    """Channel-wise Topology Refinement graph convolution (CTR-GCN style).

    A fixed, normalized base adjacency is shared by all channel groups.
    Each of the `groups` channel groups additionally learns a dynamic
    pairwise affinity from the input features (tanh of pairwise feature
    differences, temporal-mean pooled), scaled by a per-group `alpha`
    initialized to zero — so training starts as a plain GCN on the
    skeleton and topology refinement grows in as it helps.
    """

    def __init__(self, in_ch, out_ch, A_base, groups=8, rd_ch=8):
        super().__init__()
        assert out_ch % groups == 0, "out_ch must be divisible by groups"
        self.groups = groups
        self.out_ch = out_ch
        self.rd_ch = rd_ch
        self.register_buffer(
            "A_base", torch.as_tensor(A_base, dtype=torch.float32))

        self.theta = nn.Conv2d(in_ch, rd_ch * groups, 1)
        self.phi = nn.Conv2d(in_ch, rd_ch * groups, 1)
        self.value = nn.Conv2d(in_ch, out_ch, 1)
        self.alpha = nn.Parameter(torch.zeros(groups))

        self.bn = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, V)
        B, _, T, V = x.shape
        res = self.residual(x)

        # Dynamic per-group affinity from temporal-mean features
        th = self.theta(x).mean(dim=2).view(B, self.groups, self.rd_ch, V)
        ph = self.phi(x).mean(dim=2).view(B, self.groups, self.rd_ch, V)
        diff = th.unsqueeze(-1) - ph.unsqueeze(-2)        # (B, g, rd, V, V)
        refine = torch.tanh(diff).mean(dim=2)             # (B, g, V, V)

        A = self.A_base.view(1, 1, V, V) \
            + self.alpha.view(1, -1, 1, 1) * refine        # (B, g, V, V)

        v = self.value(x).view(B, self.groups, self.out_ch // self.groups, T, V)
        out = torch.einsum("bgctv,bgvw->bgctw", v, A)
        out = out.reshape(B, self.out_ch, T, V)
        return self.act(self.bn(out) + res)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestCTRGraphBlock -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_pipeline.py
git commit -m "feat: CTRGraphBlock — channel-wise topology refinement graph conv"
```

---

### Task 4: STGCNClassifier in train.py

**Files:**
- Modify: `train.py` (add after `CTRGraphBlock`; add import)
- Test: `tests/test_pipeline.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestSTGCNClassifier:
    """End-to-end CTR-GCN backbone."""

    def test_forward_shape(self):
        from train import STGCNClassifier
        model = STGCNClassifier()
        model.eval()
        x = torch.randn(2, 150, 105)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7)

    def test_zero_neighbor_no_nan(self):
        """Single-person input (zero neighbor + zero metadata) is valid."""
        from train import STGCNClassifier
        model = STGCNClassifier()
        model.eval()
        x = torch.randn(2, 150, 105)
        x[:, :, 51:] = 0
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7)
        assert not torch.isnan(out).any()

    def test_distance_affects_output(self):
        """Far neighbor (decayed) must differ from near neighbor."""
        from train import STGCNClassifier
        model = STGCNClassifier()
        model.eval()
        x_near = torch.randn(1, 150, 105)
        x_far = x_near.clone()
        x_near[:, :, 102] = 0.1
        x_far[:, :, 102] = 2.0
        with torch.no_grad():
            out_near = model(x_near)
            out_far = model(x_far)
        assert not torch.allclose(out_near, out_far, atol=1e-3)

    def test_film_conditioning(self):
        from train import STGCNClassifier
        model = STGCNClassifier(env_dim=32)
        model.eval()
        x = torch.randn(2, 150, 105)
        env_a = torch.zeros(2, 32)
        env_b = torch.ones(2, 32)
        with torch.no_grad():
            out_plain = model(x)                      # no env features
            out_a = model(x, env_features=env_a)
            out_b = model(x, env_features=env_b)
        assert out_plain.shape == (2, 7)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_param_budget(self):
        from train import STGCNClassifier
        model = STGCNClassifier()
        n = sum(p.numel() for p in model.parameters())
        assert n < 2_500_000, f"{n:,} params exceeds 2.5M budget"

    def test_backward(self):
        from train import STGCNClassifier
        model = STGCNClassifier()
        x = torch.randn(2, 150, 105, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None
        # gradient reaches both bodies and metadata
        assert x.grad[:, :, :51].abs().sum() > 0
        assert x.grad[:, :, 51:102].abs().sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestSTGCNClassifier -v`
Expected: FAIL with `ImportError: cannot import name 'STGCNClassifier'`

- [ ] **Step 3: Implement in train.py**

First, extend the graph import near the top of `train.py` (it currently imports `BONE_PAIRS` etc. from `prepare`). Add:

```python
from pose_autoresearch.graph import get_two_body_adjacency
```

Then insert after `CTRGraphBlock`:

```python
class STGCNClassifier(nn.Module):
    """CTR-GCN-style two-body skeleton classifier.

    Input contract matches PoseEventClassifier with n_bodies=2:
    (B, T, 105) = primary(51) + neighbor(51) + metadata(3), so it is a
    drop-in alternative backbone for MultiPersonPoseDataset batches.
    Selected via POSE_BACKBONE=gcn (see main()).
    """

    STAGE_CHANNELS = (64, 64, 128, 256)
    STAGE_STRIDES = (1, 1, 2, 2)

    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT, env_dim: int = 0):
        super().__init__()
        self.env_dim = env_dim
        A = get_two_body_adjacency()

        in_ch = 6  # x, y, conf for each joint + dist/rel_x/rel_y broadcast
        self.input_bn = nn.BatchNorm2d(in_ch)

        stages = []
        prev = in_ch
        for ch, stride in zip(self.STAGE_CHANNELS, self.STAGE_STRIDES):
            stages.append(nn.ModuleList([
                CTRGraphBlock(prev, ch, A),
                GraphTemporalBlock(ch, stride=stride, dropout=dropout),
            ]))
            prev = ch
        self.stages = nn.ModuleList(stages)

        if env_dim > 0:
            self.conditioners = nn.ModuleList([
                EnvironmentConditioner(env_dim, ch)
                for ch in self.STAGE_CHANNELS
            ])
        else:
            self.conditioners = None

        final_channels = self.STAGE_CHANNELS[-1]
        self.pool = TemporalAttentionPool(final_channels)
        self.fc = nn.Linear(final_channels, num_classes)

    @staticmethod
    def _gate(kps):
        """Confidence-gate xy coordinates: sigmoid(conf*5 - 2)."""
        conf = kps[:, :, :, 2:3]
        gate = torch.sigmoid(conf * 5 - 2)
        out = kps.clone()
        out[:, :, :, :2] = kps[:, :, :, :2] * gate
        return out

    def forward(self, x, env_features=None):
        """
        Args:
            x: (batch, seq_len, 105)
            env_features: optional (batch, env_dim)
        Returns:
            logits: (batch, num_classes)
        """
        B, T, _ = x.shape
        primary = x[:, :, :51].reshape(B, T, NUM_JOINTS, 3)
        neighbor = x[:, :, 51:102].reshape(B, T, NUM_JOINTS, 3)
        metadata = x[:, :, 102:105]  # (B, T, 3)

        primary = self._gate(primary)
        neighbor = self._gate(neighbor)

        # Soft distance decay on neighbor (same curve as the CNN backbone)
        decay = torch.sigmoid(5 - metadata[:, :, 0:1] * 10)  # (B, T, 1)
        neighbor = neighbor * decay.unsqueeze(-1)

        joints = torch.cat([primary, neighbor], dim=2)        # (B, T, 34, 3)
        meta = metadata.unsqueeze(2).expand(-1, -1, joints.shape[2], -1)
        feats = torch.cat([joints, meta], dim=3)              # (B, T, 34, 6)
        x = feats.permute(0, 3, 1, 2).contiguous()            # (B, 6, T, 34)

        x = self.input_bn(x)
        for i, stage in enumerate(self.stages):
            spatial, temporal = stage[0], stage[1]
            x = temporal(spatial(x))
            if self.conditioners is not None and env_features is not None:
                gamma, beta = self.conditioners[i](env_features)
                x = gamma.unsqueeze(2).unsqueeze(3) * x \
                    + beta.unsqueeze(2).unsqueeze(3)

        x = x.mean(dim=3)        # joint pool -> (B, C, T')
        x = self.pool(x)         # temporal attention -> (B, C)
        return self.fc(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestSTGCNClassifier -v`
Expected: 6 passed

- [ ] **Step 5: Run the FULL suite (regression check)**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: all tests pass (70 existing + 21 new = 91)

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_pipeline.py
git commit -m "feat: STGCNClassifier — CTR-GCN two-body backbone"
```

---

### Task 5: Backbone selection in main()

**Files:**
- Modify: `train.py` — `main()` model-construction block and the two checkpoint-path references
- Test: `tests/test_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestBackboneSelection:
    def test_build_model_cnn_default(self):
        from train import build_model, PoseEventClassifier
        model, ckpt_path = build_model("cnn", env_dim=0, n_bodies=2)
        assert isinstance(model, PoseEventClassifier)
        assert ckpt_path == "checkpoints/best_model.pt"

    def test_build_model_gcn(self):
        from train import build_model, STGCNClassifier
        model, ckpt_path = build_model("gcn", env_dim=0, n_bodies=2)
        assert isinstance(model, STGCNClassifier)
        assert ckpt_path == "checkpoints/best_model_gcn.pt"

    def test_build_model_unknown_raises(self):
        from train import build_model
        with pytest.raises(ValueError):
            build_model("transformer", env_dim=0, n_bodies=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestBackboneSelection -v`
Expected: FAIL with `ImportError: cannot import name 'build_model'`

- [ ] **Step 3: Implement build_model and wire into main()**

Add `import os` to the imports at the top of `train.py` if not already present.

Add before `main()`:

```python
def build_model(backbone, env_dim, n_bodies):
    """Construct the classifier for the requested backbone.

    Returns (model, checkpoint_path). The GCN saves to a separate
    checkpoint file so side-by-side comparison never clobbers the CNN.
    """
    if backbone == "gcn":
        model = STGCNClassifier(dropout=DROPOUT, env_dim=env_dim)
        return model, "checkpoints/best_model_gcn.pt"
    if backbone == "cnn":
        model = PoseEventClassifier(
            dropout=DROPOUT, env_dim=env_dim, n_bodies=n_bodies)
        return model, "checkpoints/best_model.pt"
    raise ValueError(f"Unknown POSE_BACKBONE: {backbone!r} (use 'cnn' or 'gcn')")
```

In `main()`, locate the current model construction:

```python
    model = PoseEventClassifier(
        dropout=DROPOUT,
        env_dim=env_dim,
        n_bodies=n_bodies,
    ).to(DEVICE)
```

Replace with:

```python
    backbone = os.environ.get("POSE_BACKBONE", "cnn").strip().lower() or "cnn"
    if backbone == "gcn" and n_bodies != 2:
        raise SystemExit(
            "POSE_BACKBONE=gcn requires multi-person data (data/splits/). "
            "Run scripts/split_data.py first.")
    model, ckpt_path = build_model(backbone, env_dim, n_bodies)
    model = model.to(DEVICE)
    print(f"Backbone: {backbone} -> {ckpt_path}")
```

Then replace the hardcoded checkpoint path in the save call:

```python
                "checkpoints/best_model.pt",
```
becomes
```python
                ckpt_path,
```

and in the final evaluation load:

```python
    ckpt = torch.load("checkpoints/best_model.pt", weights_only=True)
```
becomes
```python
    ckpt = torch.load(ckpt_path, weights_only=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestBackboneSelection -v`
Expected: 3 passed

- [ ] **Step 5: Smoke-test backbone selection (local CPU)**

Full epochs are slow on CPU with the real dataset, so verify startup only —
the goal is confirming backbone selection and shape compatibility, not training:

Run: `timeout 120 env POSE_AUTORESEARCH_MAX_TIME=5 POSE_BACKBONE=gcn .venv/bin/python train.py 2>&1 | head -25`
Expected: prints `Backbone: gcn -> checkpoints/best_model_gcn.pt` and dataset/parameter counts with no traceback. (timeout killing it mid-epoch is fine; a crash before the parameter count is not.)

Run: `timeout 120 env POSE_AUTORESEARCH_MAX_TIME=5 .venv/bin/python train.py 2>&1 | head -25`
Expected: prints `Backbone: cnn -> checkpoints/best_model.pt` — default behavior unchanged.

- [ ] **Step 6: Run the FULL suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: 94 passed

- [ ] **Step 7: Commit**

```bash
git add train.py tests/test_pipeline.py
git commit -m "feat: POSE_BACKBONE env selection with separate GCN checkpoint"
```

---

### Task 6: --backbone flag in stream_detect.py

**Files:**
- Modify: `stream_detect.py` — `StreamingDetector.__init__` (line ~482), `main()` argparse, `run_pipeline()` detector construction (line ~986)

- [ ] **Step 1: Update StreamingDetector**

Current:

```python
    def __init__(self, checkpoint_path: str, device: torch.device,
                 n_bodies: int = 1):
        self.device = device
        self.n_bodies = n_bodies
        self.model = PoseEventClassifier(n_bodies=n_bodies).to(device)
```

Replace with:

```python
    def __init__(self, checkpoint_path: str, device: torch.device,
                 n_bodies: int = 1, backbone: str = "cnn"):
        self.device = device
        self.n_bodies = n_bodies
        self.backbone = backbone
        if backbone == "gcn":
            if n_bodies != 2:
                raise ValueError(
                    "GCN backbone requires tracking mode (n_bodies=2); "
                    "remove --no-tracking or use --backbone cnn")
            from train import STGCNClassifier
            self.model = STGCNClassifier().to(device)
        else:
            self.model = PoseEventClassifier(n_bodies=n_bodies).to(device)
```

- [ ] **Step 2: Add the CLI flag in main()**

In the argparse block of `stream_detect.py` add:

```python
    parser.add_argument("--backbone", choices=["cnn", "gcn"], default="cnn",
                        help="Classifier backbone (gcn loads checkpoints/best_model_gcn.pt by default)")
```

Immediately after `args = parser.parse_args()` add:

```python
    if args.backbone == "gcn" and args.checkpoint == parser.get_default("checkpoint"):
        args.checkpoint = "checkpoints/best_model_gcn.pt"
```

- [ ] **Step 3: Wire into run_pipeline()**

Current (line ~986):

```python
    detector = StreamingDetector(args.checkpoint, device, n_bodies=n_bodies)
```

Replace with:

```python
    detector = StreamingDetector(args.checkpoint, device, n_bodies=n_bodies,
                                 backbone=args.backbone)
```

- [ ] **Step 4: Verify import + CLI**

Run: `.venv/bin/python -c "import stream_detect; print('ok')"`
Expected: `ok`

Run: `.venv/bin/python stream_detect.py --help | grep backbone`
Expected: shows the `--backbone {cnn,gcn}` option

- [ ] **Step 5: Run the FULL suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: 94 passed

- [ ] **Step 6: Commit**

```bash
git add stream_detect.py
git commit -m "feat: --backbone flag in streaming pipeline"
```

---

### Task 7: --backbone in audit script (also fixes stale checkpoint-loading bug)

**Files:**
- Modify: `scripts/audit_training_data.py` — `plot_confusion_matrix()` (line ~229) and `main()` argparse

Context: `plot_confusion_matrix` still builds `PoseEventClassifier()` with the
default 150-dim input and loads test data via single-body `PoseDataset` — it
cannot load the current 303-dim production checkpoint (this was hot-patched on
Thunder but never committed). This task fixes it properly.

- [ ] **Step 1: Fix model construction and dataset**

In `plot_confusion_matrix`, change the signature:

```python
def plot_confusion_matrix(checkpoint_path: Path, splits_dir: Path, out_path: Path):
```
becomes
```python
def plot_confusion_matrix(checkpoint_path: Path, splits_dir: Path, out_path: Path,
                          backbone: str = "cnn"):
```

Replace:

```python
    from prepare import PoseDataset, DEVICE
    from train import PoseEventClassifier
```
with
```python
    from prepare import DEVICE
    from train import PoseEventClassifier, STGCNClassifier, MultiPersonPoseDataset
```

Replace:

```python
    model = PoseEventClassifier().to(DEVICE)
```
with
```python
    if backbone == "gcn":
        model = STGCNClassifier().to(DEVICE)
    else:
        model = PoseEventClassifier(n_bodies=2).to(DEVICE)
```

Replace:

```python
    test_ds = PoseDataset(test_dir, augment=False)
```
with
```python
    test_ds = MultiPersonPoseDataset(test_dir, augment=False)
```

- [ ] **Step 2: Add the CLI flag and pass it through**

In `main()` of the audit script add:

```python
    parser.add_argument("--backbone", choices=["cnn", "gcn"], default="cnn",
                        help="Backbone of the checkpoint being audited")
```

and change the call:

```python
        plot_confusion_matrix(args.checkpoint, args.splits_dir,
                              args.output_dir / "confusion_matrix.png")
```
to
```python
        plot_confusion_matrix(args.checkpoint, args.splits_dir,
                              args.output_dir / "confusion_matrix.png",
                              backbone=args.backbone)
```

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('a','scripts/audit_training_data.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')" 2>&1 | tail -1`
Expected: `ok` (script parses; full run needs data/splits which may not exist locally)

- [ ] **Step 4: Commit**

```bash
git add scripts/audit_training_data.py
git commit -m "fix: audit script supports both backbones + 303-dim checkpoints"
```

---

### Task 8: Final verification + push

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v 2>&1 | tail -15`
Expected: 94 passed, zero failures

- [ ] **Step 2: Push**

```bash
git push origin HEAD
```

- [ ] **Step 3: Report Thunder commands for the comparison runs**

The Thunder comparison protocol (run manually after push):

```bash
git pull origin main
# Control (CNN):
POSE_AUTORESEARCH_MAX_TIME=3600 python train.py
# Challenger (GCN):
POSE_AUTORESEARCH_MAX_TIME=3600 POSE_BACKBONE=gcn python train.py
# GCN confusion matrix:
python scripts/audit_training_data.py --checkpoint checkpoints/best_model_gcn.pt --backbone gcn --quick
```

Decision rule: winner needs higher test accuracy AND fall recall >= the CNN's
98.37%. Loser's checkpoint is archived under experiments/.
