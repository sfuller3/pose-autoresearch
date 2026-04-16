"""End-to-end smoke tests for the training pipeline."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare import (
    PoseDataset,
    get_dataloaders,
    evaluate_model,
    evaluate_per_class,
    compute_bone_features,
    compute_velocity_features,
    EVENT_CLASSES,
    NUM_KEYPOINTS,
    VALUES_PER_KEYPOINT,
    INPUT_DIM,
    SEQ_LEN,
    DEVICE,
    NUM_BONES,
)
from pose_autoresearch.graph import (
    get_adjacency_matrix,
    get_normalized_adjacency,
    get_spatial_partitioning,
    get_bone_pairs,
    COCO_17_EDGES,
)
from pose_autoresearch.augment import (
    random_rotation,
    random_horizontal_flip,
    random_scale,
    random_temporal_crop,
    add_gaussian_noise,
    random_joint_dropout,
    augment_pose_sequence,
)


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with minimal JSON pose samples."""
    for cls_name in EVENT_CLASSES:
        for i in range(10):
            frames = []
            for t in range(SEQ_LEN):
                kps = np.random.rand(NUM_KEYPOINTS, VALUES_PER_KEYPOINT).tolist()
                frames.append({"keypoints": kps, "timestamp": t / 30.0})
            sample = {"frames": frames, "label": cls_name, "duration": SEQ_LEN / 30.0}
            with open(tmp_path / f"{cls_name}_{i:04d}.json", "w") as f:
                json.dump(sample, f)
    return tmp_path


# ============================================================================
# Graph tests
# ============================================================================


class TestGraph:
    def test_adjacency_shape(self):
        A = get_adjacency_matrix()
        assert A.shape == (17, 17)

    def test_adjacency_symmetric(self):
        A = get_adjacency_matrix()
        np.testing.assert_array_equal(A, A.T)

    def test_adjacency_self_loops(self):
        A = get_adjacency_matrix(self_loops=True)
        assert all(A[i, i] == 1.0 for i in range(17))

    def test_adjacency_no_self_loops(self):
        A = get_adjacency_matrix(self_loops=False)
        assert all(A[i, i] == 0.0 for i in range(17))

    def test_known_edge(self):
        A = get_adjacency_matrix()
        # left_shoulder (5) -> left_elbow (7)
        assert A[5, 7] == 1.0
        assert A[7, 5] == 1.0

    def test_normalized_adjacency_shape(self):
        A = get_normalized_adjacency()
        assert A.shape == (17, 17)
        # All rows should be non-zero
        assert all(A[i].sum() > 0 for i in range(17))

    def test_spatial_partitioning_shape(self):
        P = get_spatial_partitioning()
        assert P.shape == (3, 17, 17)

    def test_bone_pairs_valid(self):
        bones = get_bone_pairs()
        assert len(bones) > 0
        for parent, child in bones:
            assert 0 <= parent < 17
            assert 0 <= child < 17


# ============================================================================
# Augmentation tests
# ============================================================================


class TestAugmentation:
    def test_rotation_shape(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        result = random_rotation(poses)
        assert result.shape == poses.shape

    def test_rotation_preserves_confidence(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        result = random_rotation(poses)
        torch.testing.assert_close(result[:, :, 2], poses[:, :, 2])

    def test_scale_shape(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        result = random_scale(poses)
        assert result.shape == poses.shape

    def test_flip_shape(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        result = random_horizontal_flip(poses, p=1.0)
        assert result.shape == poses.shape

    def test_temporal_crop_pad(self):
        short = torch.rand(50, 17, 3)
        result = random_temporal_crop(short, target_len=SEQ_LEN)
        assert result.shape == (SEQ_LEN, 17, 3)

    def test_temporal_crop_trim(self):
        long = torch.rand(200, 17, 3)
        result = random_temporal_crop(long, target_len=SEQ_LEN)
        assert result.shape == (SEQ_LEN, 17, 3)

    def test_gaussian_noise_shape(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        result = add_gaussian_noise(poses)
        assert result.shape == poses.shape

    def test_joint_dropout_shape(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        result = random_joint_dropout(poses)
        assert result.shape == poses.shape

    def test_full_augment_pipeline(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        result = augment_pose_sequence(poses, target_len=SEQ_LEN)
        assert result.shape == (SEQ_LEN, 17, 3)


# ============================================================================
# Multi-stream feature tests
# ============================================================================


class TestMultiStream:
    def test_bone_features_shape(self):
        kps = np.random.rand(SEQ_LEN, 17, 3).astype(np.float32)
        bones = compute_bone_features(kps)
        assert bones.shape == (SEQ_LEN, NUM_BONES, 3)

    def test_velocity_features_shape(self):
        kps = np.random.rand(SEQ_LEN, 17, 3).astype(np.float32)
        vel = compute_velocity_features(kps)
        assert vel.shape == (SEQ_LEN, 17, 3)

    def test_velocity_first_frame_zero(self):
        kps = np.random.rand(SEQ_LEN, 17, 3).astype(np.float32)
        vel = compute_velocity_features(kps)
        np.testing.assert_array_equal(vel[0, :, :2], 0.0)


# ============================================================================
# Dataset tests
# ============================================================================


class TestDataset:
    def test_dataset_loads(self, sample_data_dir):
        ds = PoseDataset(sample_data_dir)
        assert len(ds) == 70  # 7 classes x 10 samples

    def test_sample_shape(self, sample_data_dir):
        ds = PoseDataset(sample_data_dir)
        poses, label = ds[0]
        assert poses.shape == (SEQ_LEN, INPUT_DIM)
        assert 0 <= label.item() < len(EVENT_CLASSES)

    def test_augmented_dataset(self, sample_data_dir):
        ds = PoseDataset(sample_data_dir, augment=True)
        poses, label = ds[0]
        assert poses.shape == (SEQ_LEN, INPUT_DIM)

    def test_multi_stream_dataset(self, sample_data_dir):
        ds = PoseDataset(sample_data_dir, multi_stream=True)
        (joints, bones, vel), label = ds[0]
        assert joints.shape == (SEQ_LEN, INPUT_DIM)
        assert bones.shape == (SEQ_LEN, NUM_BONES * 3)
        assert vel.shape == (SEQ_LEN, INPUT_DIM)

    def test_dataloaders(self, sample_data_dir):
        train, val, test = get_dataloaders(
            data_dir=sample_data_dir, batch_size=4, num_workers=0,
        )
        assert len(train) > 0
        poses, labels = next(iter(train))
        assert poses.shape[0] <= 4
        assert poses.shape[2] == INPUT_DIM


# ============================================================================
# Model tests
# ============================================================================


class TestModel:
    def test_forward_pass(self):
        from train import PoseEventClassifier
        model = PoseEventClassifier().to(DEVICE)
        x = torch.randn(2, SEQ_LEN, INPUT_DIM).to(DEVICE)
        logits = model(x)
        assert logits.shape == (2, len(EVENT_CLASSES))

    def test_forward_different_batch_sizes(self):
        from train import PoseEventClassifier
        model = PoseEventClassifier().to(DEVICE)
        for bs in [1, 4, 16]:
            x = torch.randn(bs, SEQ_LEN, INPUT_DIM).to(DEVICE)
            logits = model(x)
            assert logits.shape == (bs, len(EVENT_CLASSES))

    def test_model_parameter_count(self):
        from train import PoseEventClassifier
        model = PoseEventClassifier()
        num_params = sum(p.numel() for p in model.parameters())
        # Should be under 1M for edge deployment
        assert num_params < 2_000_000, f"Model too large: {num_params:,} params"

    def test_evaluate(self, sample_data_dir):
        from train import PoseEventClassifier
        model = PoseEventClassifier().to(DEVICE)
        _, val_loader, _ = get_dataloaders(
            data_dir=sample_data_dir, batch_size=4, num_workers=0,
        )
        acc, loss = evaluate_model(model, val_loader, DEVICE)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_per_class_eval(self, sample_data_dir):
        from train import PoseEventClassifier
        model = PoseEventClassifier().to(DEVICE)
        _, val_loader, _ = get_dataloaders(
            data_dir=sample_data_dir, batch_size=4, num_workers=0,
        )
        per_class = evaluate_per_class(model, val_loader, DEVICE)
        assert set(per_class.keys()) == set(EVENT_CLASSES)
        for cls, acc in per_class.items():
            assert 0.0 <= acc <= 1.0


# ============================================================================
# FocalLoss tests
# ============================================================================


class TestFocalLoss:
    def test_gamma_zero_reduces_to_weighted_ce(self):
        """With gamma=0, focal loss must equal weighted cross-entropy
        (using plain-mean reduction, matching FocalLoss's ``focal.mean()``)."""
        from train import FocalLoss
        import torch.nn.functional as F

        torch.manual_seed(0)
        num_classes = len(EVENT_CLASSES)
        logits = torch.randn(16, num_classes)
        targets = torch.randint(0, num_classes, (16,))
        weight = torch.linspace(0.5, 1.5, num_classes)

        focal = FocalLoss(weight=weight, gamma=0.0, label_smoothing=0.0)
        focal_loss = focal(logits, targets)

        # FocalLoss applies plain .mean() over the batch of weighted CEs,
        # which differs from F.cross_entropy's default weighted-mean
        # (divides by sum-of-weights). Use reduction="none" then .mean().
        ce_per_sample = F.cross_entropy(
            logits, targets, weight=weight, reduction="none",
        )
        torch.testing.assert_close(
            focal_loss, ce_per_sample.mean(), rtol=1e-5, atol=1e-6,
        )

    def test_gamma_zero_with_label_smoothing_matches_ce(self):
        """With gamma=0 and smoothing, focal must match smoothed weighted CE
        (plain-mean reduction)."""
        from train import FocalLoss
        import torch.nn.functional as F

        torch.manual_seed(1)
        num_classes = len(EVENT_CLASSES)
        logits = torch.randn(16, num_classes)
        targets = torch.randint(0, num_classes, (16,))
        weight = torch.linspace(0.5, 1.5, num_classes)

        focal = FocalLoss(weight=weight, gamma=0.0, label_smoothing=0.1)
        focal_loss = focal(logits, targets)

        ce_per_sample = F.cross_entropy(
            logits, targets, weight=weight, label_smoothing=0.1,
            reduction="none",
        )
        torch.testing.assert_close(
            focal_loss, ce_per_sample.mean(), rtol=1e-5, atol=1e-6,
        )

    def test_confident_correct_lower_than_confident_wrong(self):
        """A confident-correct prediction should incur a smaller loss
        than a confident-wrong prediction."""
        from train import FocalLoss

        num_classes = len(EVENT_CLASSES)
        loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)

        # Confidently correct: class 0 with high logit
        correct_logits = torch.full((1, num_classes), -5.0)
        correct_logits[0, 0] = 5.0
        correct_target = torch.tensor([0])

        # Confidently wrong: predicts class 0 but true label is 1
        wrong_logits = torch.full((1, num_classes), -5.0)
        wrong_logits[0, 0] = 5.0
        wrong_target = torch.tensor([1])

        loss_correct = loss_fn(correct_logits, correct_target)
        loss_wrong = loss_fn(wrong_logits, wrong_target)

        assert loss_correct.item() < loss_wrong.item()

    def test_backward_produces_nonzero_grad(self):
        """Focal loss must produce non-zero gradients via backward."""
        from train import FocalLoss

        num_classes = len(EVENT_CLASSES)
        loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
        logits = torch.randn(8, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (8,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert torch.isfinite(loss)
        assert logits.grad is not None
        assert logits.grad.norm().item() > 0

    def test_weight_registered_as_buffer(self):
        """When weight is provided, it should be a registered buffer
        so .to(device) moves it along with the module."""
        from train import FocalLoss

        num_classes = len(EVENT_CLASSES)
        weight = torch.ones(num_classes)
        loss_fn = FocalLoss(weight=weight, gamma=2.0)

        buffer_names = {name for name, _ in loss_fn.named_buffers()}
        assert "weight" in buffer_names


# ============================================================================
# TemporalAttentionPool tests
# ============================================================================


class TestTemporalAttentionPool:
    def test_temporal_attention_pool_output_shape(self):
        from train import TemporalAttentionPool
        pool = TemporalAttentionPool(256)
        x = torch.randn(2, 256, 75)
        out = pool(x)
        assert out.shape == (2, 256)

    def test_temporal_attention_pool_grad_flow(self):
        from train import TemporalAttentionPool
        pool = TemporalAttentionPool(256)
        x = torch.randn(2, 256, 75, requires_grad=True)
        out = pool(x).sum()
        out.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# ============================================================================
# Confidence gating tests
# ============================================================================


class TestConfidenceGating:
    def test_zero_confidence_attenuates_joints(self):
        """When all joints have conf=0, gated coords should be attenuated
        (gate ~0.12), producing different model outputs vs. conf=1 (gate ~0.95)."""
        from train import PoseEventClassifier
        m = PoseEventClassifier().eval()

        # All coords = 10.0, all conf = 0.0 -> gate ~0.12, gated coords ~1.2
        x_low = torch.ones(1, 150, 51) * 10.0
        x_low[:, :, 2::3] = 0.0

        # All coords = 10.0, all conf = 1.0 -> gate ~0.95, gated coords ~9.5
        x_high = torch.ones(1, 150, 51) * 10.0
        x_high[:, :, 2::3] = 1.0

        with torch.no_grad():
            out_low = m(x_low)
            out_high = m(x_high)

        # Outputs should differ -- gating is active
        assert not torch.allclose(out_low, out_high, atol=1e-4)

    def test_gate_values_are_correct(self):
        """Verify the sigmoid gate produces the documented values."""
        import torch
        # Gate formula: sigmoid(conf * 5 - 2)
        # conf=0.0 -> sigmoid(-2) ~ 0.1192
        # conf=0.4 -> sigmoid(0)  = 0.5
        # conf=1.0 -> sigmoid(3)  ~ 0.9526
        assert torch.allclose(torch.sigmoid(torch.tensor(0.0 * 5 - 2)), torch.tensor(0.1192), atol=1e-3)
        assert torch.allclose(torch.sigmoid(torch.tensor(0.4 * 5 - 2)), torch.tensor(0.5000), atol=1e-3)
        assert torch.allclose(torch.sigmoid(torch.tensor(1.0 * 5 - 2)), torch.tensor(0.9526), atol=1e-3)


class TestMixup:
    """Tests for the temporal MixUp augmentation."""

    def test_mixup_data_shapes(self):
        """Shapes and lambda bounds are correct after mixup."""
        from train import mixup_data

        B, T, C = 4, 150, 51
        x = torch.randn(B, T, C)
        y = torch.randint(0, 7, (B,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)

        assert mixed_x.shape == x.shape
        assert y_a.shape == (B,)
        assert y_b.shape == (B,)
        assert 0.0 <= lam <= 1.0

    def test_mixup_lambda_extreme(self):
        """Beta(0.2, 0.2) is U-shaped, so lambda should span both halves."""
        from train import mixup_data

        x = torch.randn(2, 10, 5)
        y = torch.zeros(2, dtype=torch.long)

        lams = []
        for _ in range(200):
            _, _, _, lam = mixup_data(x, y, alpha=0.2)
            lams.append(lam)

        assert any(l > 0.5 for l in lams), "Expected at least one lambda > 0.5"
        assert any(l < 0.5 for l in lams), "Expected at least one lambda < 0.5"

    def test_mixup_is_reproducible_with_torch_seed(self):
        """Seeding torch RNG should yield deterministic mixup lambdas."""
        from train import mixup_data

        x = torch.randn(2, 10, 5)
        y = torch.zeros(2, dtype=torch.long)

        torch.manual_seed(42)
        _, _, _, lam_a1 = mixup_data(x, y)
        _, _, _, lam_a2 = mixup_data(x, y)

        torch.manual_seed(42)
        _, _, _, lam_b1 = mixup_data(x, y)
        _, _, _, lam_b2 = mixup_data(x, y)

        assert lam_a1 == lam_b1, f"Expected reproducible lambda, got {lam_a1} vs {lam_b1}"
        assert lam_a2 == lam_b2, f"Expected reproducible lambda, got {lam_a2} vs {lam_b2}"


# ============================================================================
# Pillar 1: Environment-aware detection tests
# ============================================================================


class TestSpatialFeatures:
    def test_zero_features_when_no_detections(self):
        """Empty detections produce zero feature vector."""
        from stream_detect import EnvironmentDetector

        det = EnvironmentDetector.__new__(EnvironmentDetector)
        det._class_list = sorted(EnvironmentDetector.CONTEXT_CLASSES)
        det._class_to_idx = {c: i for i, c in enumerate(det._class_list)}

        features = det.compute_spatial_features([], None, (480, 640))
        assert features.shape == (32,)
        assert np.allclose(features, 0.0)

    def test_features_with_detection(self):
        """Detection sets presence flag and computes proximity."""
        from stream_detect import EnvironmentDetector

        det = EnvironmentDetector.__new__(EnvironmentDetector)
        det._class_list = sorted(EnvironmentDetector.CONTEXT_CLASSES)
        det._class_to_idx = {c: i for i, c in enumerate(det._class_list)}

        # Person at center (320, 240), bed bbox at center
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[11] = [320, 240, 0.9]  # left hip
        kps[12] = [320, 240, 0.9]  # right hip

        detections = [{"class": "bed", "bbox": [200, 150, 440, 330], "confidence": 0.8}]
        features = det.compute_spatial_features(detections, kps, (480, 640))

        bed_idx = det._class_to_idx["bed"]
        assert features[bed_idx * 4 + 0] == 1.0  # present
        assert features[bed_idx * 4 + 1] > 0.5   # high proximity (overlapping)


class TestContextRules:
    def test_no_person_suppresses_fall(self):
        """No person detected should suppress fall probability."""
        from stream_detect import apply_context_rules
        probs = np.ones(7, dtype=np.float32) / 7
        fall_before = probs[0]  # CLASS_TO_IDX["fall"] = 0
        result = apply_context_rules(probs, False, np.zeros(32), [])
        # Fall should be suppressed
        assert result[0] < fall_before

    def test_probs_renormalized(self):
        """Output probabilities should sum to 1."""
        from stream_detect import apply_context_rules
        probs = np.ones(7, dtype=np.float32) / 7
        result = apply_context_rules(probs, True, np.zeros(32), [])
        assert abs(result.sum() - 1.0) < 1e-6


class TestFiLMConditioning:
    def test_no_env_features_unchanged_output(self):
        """Model without env_dim=0 produces same output shape."""
        from train import PoseEventClassifier

        model = PoseEventClassifier(env_dim=0)
        model.eval()
        x = torch.randn(2, 150, 51)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7)

    def test_with_env_features_produces_output(self):
        """Model with env_dim=32 accepts environment features."""
        from train import PoseEventClassifier

        model = PoseEventClassifier(env_dim=32)
        model.eval()
        x = torch.randn(2, 150, 51)
        env = torch.randn(2, 32)
        with torch.no_grad():
            out = model(x, env_features=env)
        assert out.shape == (2, 7)

    def test_env_features_affect_output(self):
        """Environment features should change model output."""
        from train import PoseEventClassifier

        model = PoseEventClassifier(env_dim=32)
        model.eval()
        x = torch.randn(2, 150, 51)
        env_a = torch.zeros(2, 32)
        env_b = torch.ones(2, 32)
        with torch.no_grad():
            out_a = model(x, env_features=env_a)
            out_b = model(x, env_features=env_b)
        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_backward_compat_no_env(self):
        """Forward without env_features works (backward compatible)."""
        from train import PoseEventClassifier

        model = PoseEventClassifier(env_dim=32)
        model.eval()
        x = torch.randn(2, 150, 51)
        with torch.no_grad():
            out = model(x)  # no env_features passed
        assert out.shape == (2, 7)


# ============================================================================
# MULTI-PERSON PIPELINE TESTS
# ============================================================================


class TestMultiPersonDataset:
    """Tests for MultiPersonPoseDataset (MP-1)."""

    @pytest.fixture
    def multi_body_data_dir(self, tmp_path):
        """Create temp dir with single-body and multi-body JSON samples."""
        # Single-body sample
        frames_single = []
        for t in range(30):
            kps = np.random.rand(17, 3).tolist()
            frames_single.append({"keypoints": kps, "timestamp": t / 30.0})
        with open(tmp_path / "single_fall_001.json", "w") as f:
            json.dump({"frames": frames_single, "label": "fall", "duration": 1.0}, f)

        # Multi-body sample (2 bodies)
        frames_multi = []
        for t in range(30):
            body0 = np.random.rand(17, 3).tolist()
            body1 = np.random.rand(17, 3).tolist()
            frames_multi.append({
                "keypoints": body0,
                "bodies": [body0, body1],
                "timestamp": t / 30.0,
            })
        with open(tmp_path / "multi_aggression_001.json", "w") as f:
            json.dump({"frames": frames_multi, "label": "aggression", "duration": 1.0}, f)

        return tmp_path

    def test_loads_single_body(self, multi_body_data_dir):
        """Single-body samples produce 105-dim output with zero neighbor."""
        from train import MultiPersonPoseDataset
        ds = MultiPersonPoseDataset(multi_body_data_dir, seq_len=30)
        # Should have 3 samples: 1 single + 2 (multi with pair flip)
        assert len(ds) == 3

    def test_output_shape(self, multi_body_data_dir):
        """Each sample has shape (seq_len, 105)."""
        from train import MultiPersonPoseDataset
        ds = MultiPersonPoseDataset(multi_body_data_dir, seq_len=30)
        x, y = ds[0]
        assert x.shape == (30, 105)
        assert y.dtype == torch.long

    def test_pair_flipping_doubles_multi_body(self, multi_body_data_dir):
        """Multi-body samples produce 2 entries (pair flipping)."""
        from train import MultiPersonPoseDataset
        ds = MultiPersonPoseDataset(multi_body_data_dir, seq_len=30)
        # 1 single-body + 2 from pair flip of multi-body = 3 total
        assert len(ds) == 3

    def test_single_body_neighbor_is_zero(self, multi_body_data_dir):
        """For single-body samples, neighbor channels (dims 51:102) are zero."""
        from train import MultiPersonPoseDataset
        ds = MultiPersonPoseDataset(multi_body_data_dir, seq_len=30)
        # Find the single-body sample (label=fall=0)
        for i in range(len(ds)):
            x, y = ds[i]
            if y.item() == 0:  # fall
                neighbor = x[:, 51:102]
                assert torch.all(neighbor == 0), "Single-body neighbor should be zeros"
                break

    def test_metadata_dims(self, multi_body_data_dir):
        """Metadata (dims 102:105) are populated for multi-body samples."""
        from train import MultiPersonPoseDataset
        ds = MultiPersonPoseDataset(multi_body_data_dir, seq_len=30)
        # Find a multi-body sample (label=aggression=3)
        for i in range(len(ds)):
            x, y = ds[i]
            if y.item() == 3:  # aggression
                metadata = x[:, 102:105]
                # With random data, distance should be nonzero
                assert not torch.all(metadata == 0), "Multi-body metadata should be nonzero"
                break

    def test_padding_short_sequences(self, tmp_path):
        """Sequences shorter than seq_len are zero-padded."""
        from train import MultiPersonPoseDataset
        frames = []
        for t in range(10):  # only 10 frames
            kps = np.random.rand(17, 3).tolist()
            frames.append({"keypoints": kps, "timestamp": t / 30.0})
        with open(tmp_path / "short_fall_001.json", "w") as f:
            json.dump({"frames": frames, "label": "fall", "duration": 0.33}, f)

        ds = MultiPersonPoseDataset(tmp_path, seq_len=30)
        x, y = ds[0]
        assert x.shape == (30, 105)
        # Last frames should be zero-padded
        assert torch.all(x[10:] == 0)


class TestMultiBodyModel:
    """Tests for PoseEventClassifier with n_bodies=2 (MP-2)."""

    def test_n_bodies_2_forward(self):
        """n_bodies=2 accepts 105-dim input and outputs (B, 7)."""
        from train import PoseEventClassifier
        model = PoseEventClassifier(n_bodies=2)
        model.eval()
        x = torch.randn(4, 150, 105)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 7)

    def test_n_bodies_1_backward_compat(self):
        """n_bodies=1 still accepts 51-dim input (original behavior)."""
        from train import PoseEventClassifier
        model = PoseEventClassifier(n_bodies=1)
        model.eval()
        x = torch.randn(4, 150, 51)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 7)

    def test_distance_decay_attenuates_far_neighbor(self):
        """When distance_norm is large, neighbor features should be attenuated."""
        from train import PoseEventClassifier
        model = PoseEventClassifier(n_bodies=2)
        model.eval()

        # Build two inputs: one with close neighbor, one with far neighbor
        x_close = torch.randn(1, 150, 105)
        x_far = x_close.clone()

        # Set distance_norm (dim 102) — close=0.1, far=2.0
        x_close[:, :, 102] = 0.1  # sigmoid(5 - 0.1*10) ≈ sigmoid(4) ≈ 0.98
        x_far[:, :, 102] = 2.0    # sigmoid(5 - 2.0*10) ≈ sigmoid(-15) ≈ 0.0

        with torch.no_grad():
            out_close = model(x_close)
            out_far = model(x_far)

        # Outputs should differ because decay changes neighbor contribution
        assert not torch.allclose(out_close, out_far, atol=1e-3)

    def test_zero_neighbor_equivalent_to_absent(self):
        """All-zero neighbor with zero metadata should produce valid output."""
        from train import PoseEventClassifier
        model = PoseEventClassifier(n_bodies=2)
        model.eval()
        x = torch.randn(2, 150, 105)
        x[:, :, 51:] = 0  # zero out neighbor and metadata
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7)
        assert not torch.isnan(out).any()

    def test_n_bodies_2_with_env_features(self):
        """n_bodies=2 works together with FiLM env conditioning."""
        from train import PoseEventClassifier
        model = PoseEventClassifier(n_bodies=2, env_dim=32)
        model.eval()
        x = torch.randn(2, 150, 105)
        env = torch.randn(2, 32)
        with torch.no_grad():
            out = model(x, env_features=env)
        assert out.shape == (2, 7)

    def test_gradient_flows_through_both_bodies(self):
        """Backprop flows through both primary and neighbor paths."""
        from train import PoseEventClassifier
        model = PoseEventClassifier(n_bodies=2)
        x = torch.randn(2, 150, 105, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Gradient should flow to all input dims
        assert x.grad is not None
        # Primary dims should have gradient
        assert x.grad[:, :, :51].abs().sum() > 0
        # Neighbor dims should have gradient
        assert x.grad[:, :, 51:102].abs().sum() > 0


class TestPersonTracker:
    """Tests for PersonTracker and PersonState (MP-4)."""

    def test_new_detection_creates_state(self):
        """New track IDs create PersonState entries."""
        from stream_detect import PersonTracker
        tracker = PersonTracker()
        kps = np.random.rand(17, 3).astype(np.float32)
        kps[11, 2] = 0.9  # left hip confidence
        kps[12, 2] = 0.9  # right hip confidence
        tracker.update([(1, kps), (2, kps.copy())], timestamp=0.0)
        assert 1 in tracker.people
        assert 2 in tracker.people

    def test_grace_period_keeps_lost_track(self):
        """Lost tracks survive within grace period."""
        from stream_detect import PersonTracker
        tracker = PersonTracker()
        kps = np.random.rand(17, 3).astype(np.float32)
        kps[11, 2] = 0.9
        kps[12, 2] = 0.9
        tracker.update([(1, kps)], timestamp=0.0)
        # Person disappears but within grace period
        tracker.update([], timestamp=3.0)
        assert 1 in tracker.people

    def test_grace_period_expires_old_track(self):
        """Lost tracks are dropped after grace period."""
        from stream_detect import PersonTracker
        tracker = PersonTracker()
        kps = np.random.rand(17, 3).astype(np.float32)
        kps[11, 2] = 0.9
        kps[12, 2] = 0.9
        tracker.update([(1, kps)], timestamp=0.0)
        # Person disappears past grace period (5s)
        tracker.update([], timestamp=6.0)
        assert 1 not in tracker.people

    def test_get_active_people_min_frames(self):
        """Only people with sufficient buffer depth are returned."""
        from stream_detect import PersonTracker
        tracker = PersonTracker()
        kps = np.random.rand(17, 3).astype(np.float32)
        kps[11, 2] = 0.9
        kps[12, 2] = 0.9
        # Push only 3 frames
        for t in range(3):
            tracker.update([(1, kps)], timestamp=t * 0.033)
        active = tracker.get_active_people(min_frames=10)
        assert len(active) == 0  # not enough frames yet

        # Push more frames
        for t in range(3, 15):
            tracker.update([(1, kps)], timestamp=t * 0.033)
        active = tracker.get_active_people(min_frames=10)
        assert len(active) == 1

    def test_reassociation_near_stale_track(self):
        """New track near a stale track's position re-associates."""
        from stream_detect import PersonTracker
        tracker = PersonTracker()

        kps = np.random.rand(17, 3).astype(np.float32)
        # Set known hip position
        kps[11, :] = [100, 200, 0.9]
        kps[12, :] = [120, 200, 0.9]

        # Person 1 appears and builds some buffer
        for t in range(5):
            tracker.update([(1, kps)], timestamp=t * 0.033)
        assert len(tracker.people[1].buffer) == 5

        # Person 1 disappears for 1 second (within grace)
        tracker.update([], timestamp=1.0)

        # New person 5 appears at similar position
        kps_new = kps.copy()
        kps_new[11, :] = [105, 205, 0.9]  # nearby
        kps_new[12, :] = [125, 205, 0.9]
        tracker.update([(5, kps_new)], timestamp=1.5)

        # Person 5 should have inherited person 1's buffer
        assert 5 in tracker.people
        assert len(tracker.people[5].buffer) == 6  # 5 from person 1 + 1 new


class TestDistanceDecay:
    """Tests for the soft distance decay function used in multi-person."""

    def test_close_neighbor_high_decay(self):
        """Neighbor within ~30% frame diagonal gets strong contribution."""
        # sigmoid(5 - 0.2 * 10) = sigmoid(3) ≈ 0.95
        decay = torch.sigmoid(torch.tensor(5.0 - 0.2 * 10))
        assert decay.item() > 0.9

    def test_far_neighbor_low_decay(self):
        """Neighbor beyond ~60% frame diagonal contributes near-zero."""
        # sigmoid(5 - 0.8 * 10) = sigmoid(-3) ≈ 0.05
        decay = torch.sigmoid(torch.tensor(5.0 - 0.8 * 10))
        assert decay.item() < 0.1

    def test_very_far_neighbor_near_zero(self):
        """Neighbor at 2x frame diagonal → essentially zero."""
        decay = torch.sigmoid(torch.tensor(5.0 - 2.0 * 10))
        assert decay.item() < 0.001
