"""Offline tests. No downloads, no GPU.

Several tests here are written specifically to catch defects found by the
referee pass, each of which passed a weaker version of this suite:

  * `test_mcnemar_survives_large_discordant_counts` -- the exact tail used
    `2.0 ** n`, which overflows at n >= 1024.
  * `test_paired_gap_se_is_not_clamped_to_zero` -- the covariance term omitted
    `+2ab`, so the function returned exactly 0.0 for almost every input.
  * `test_paired_gap_se_rejects_impossible_agreement` -- the old code clamped a
    negative variance to 0.0, turning a bug into a plausible-looking answer.
  * `test_selection_rules_differ_from_argmax_when_they_should` -- the removed
    `paired` rule provably equalled `argmax`; the suite never noticed.
  * `test_splits_are_disjoint_and_sized` -- the disjointness body was skipped
    entirely because the fake dataset returned `indices=None`.
  * `test_floor_rows_are_like_for_like` -- guards against re-introducing a
    dispersion plotted against a threshold.
"""
from __future__ import annotations

import numpy as np
import torch

from src.config import load_config
from src.data.datasets import (FakeImages, _stratified_pool_indices, build_datasets,
                               build_loaders, eval_transform)
from src.engine.averaging import average_state_dicts, top_q_epoch_paths
from src.engine.selector import (replay_all_rules, replay_select, select_argmax, select_one_se,
                                 stack_correctness, validation_gap_dispersion)
from src.engine.trainer import correctness_matrix, floor_stop_epoch, pool_loss
from src.losses.build import build_loss
from src.models.factory import build_model, count_parameters
from src.stats.inference import (gap_for_correct_ranking, mcnemar_exact, minimum_detectable_effect,
                                 paired_gap_se, paired_test, selection_floor_table,
                                 standard_error, validation_gap_se)


def _corr(n_epochs: int, n_examples: int, accs: list[float], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_epochs, n_examples)) < np.asarray(accs)[:, None]


def test_config_loads() -> None:
    cfg = load_config("configs/smoke.yaml")
    assert cfg.data.dataset == "fake"
    assert cfg.model.num_classes == 10


def test_splits_are_disjoint_and_sized() -> None:
    """The study's central invariant. Previously this body never executed."""
    cfg = load_config("configs/smoke.yaml")
    sets = build_datasets(cfg)
    indices = sets["indices"]
    assert indices is not None, "fake dataset must return indices so this test runs"
    fit = set(indices.fit.tolist())
    pool = set(indices.pool.tolist())
    assert not (fit & pool), "fit and pool index the same array and must be disjoint"
    assert len(fit) == len(sets["fit"])
    assert len(pool) == len(sets["pool"])
    # `test` indexes a separate array, so cross-split index disjointness is not a
    # meaningful check; the invariant that matters there is the transform, which
    # must be the evaluation transform with no augmentation.
    assert len(indices.test) == len(sets["test"])
    test_image, _ = sets["test"][0]
    pool_image, _ = sets["pool"][0]
    assert test_image.shape == pool_image.shape
    assert sets["fit"].transform.transforms[0].p == 0.5, "fit may flip; pool must not"
    assert type(sets["pool"].transform.transforms[0]).__name__ == "ToTensor", \
        "pool must not use the training transform's augmentation"


def test_stratified_split_is_disjoint_and_respects_fraction() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(10), 100)
    rng.shuffle(labels)
    fit, pool = _stratified_pool_indices(labels, 0.2, seed=11)
    assert not (set(fit.tolist()) & set(pool.tolist()))
    assert fit.size + pool.size == labels.size
    per_class_pool = np.bincount(labels[pool], minlength=10)
    per_class_total = np.bincount(labels, minlength=10)
    assert (per_class_pool >= 1).all()
    assert np.allclose(per_class_pool / per_class_total, 0.2, atol=0.02)


def test_models_forward_and_count() -> None:
    for in_channels in (1, 3):
        model = build_model("simplecnn", 10, in_channels)
        out = model(torch.randn(4, in_channels, 28, 28))
        assert out.shape == (4, 10)
        assert count_parameters(model) > 0


def test_losses_finite() -> None:
    cfg = load_config("configs/smoke.yaml")
    logits = torch.randn(16, 10)
    target = torch.randint(0, 10, (16,))
    assert torch.isfinite(build_loss(cfg)(logits, target))


def test_correctness_and_pool_loss_shapes() -> None:
    cfg = load_config("configs/smoke.yaml")
    loaders = build_loaders(cfg, build_datasets(cfg))
    model = build_model("simplecnn", 10, 1)
    matrix = correctness_matrix(model, loaders["pool"], torch.device("cpu"))
    assert matrix.shape == (len(loaders["pool"].dataset),)
    assert matrix.dtype == np.bool_
    loss = pool_loss(model, loaders["pool"], torch.device("cpu"))
    assert np.isfinite(loss) and loss > 0


def test_floor_functions_are_distinct_and_ordered() -> None:
    se = validation_gap_se(0.95, 1000)
    ranking = gap_for_correct_ranking(0.95, 1000)
    mde = minimum_detectable_effect(0.95, 1000)
    assert se < ranking < mde
    assert abs(ranking - 1.6448536269514722 * se) < 1e-9
    assert abs(mde - (1.959963984540054 + 1.6448536269514722) * se) < 1e-9


def test_mde_scales_as_inverse_square_root() -> None:
    sizes = (250, 500, 1000, 5000)
    reference = minimum_detectable_effect(0.92, sizes[0])
    for n in sizes[1:]:
        assert abs(minimum_detectable_effect(0.92, n) / (reference * (sizes[0] / n) ** 0.5) - 1.0) < 1e-9


def test_selection_floor_table_is_generated_not_transcribed() -> None:
    rows = selection_floor_table(0.95, (1000,))
    assert rows[0]["n_val"] == 1000
    assert abs(rows[0]["mde"] - minimum_detectable_effect(0.95, 1000)) < 1e-12


def test_mcnemar_edges_and_overflow() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) < 0.1
    assert mcnemar_exact(25, 25) > 0.5
    # Previously raised OverflowError because of `2.0 ** n`.
    value = mcnemar_exact(700, 700)
    assert 0.0 <= value <= 1.0


def test_paired_gap_se_is_not_clamped_to_zero() -> None:
    """The old covariance dropped +2ab and clamped negatives, always returning 0.0."""
    assert abs(paired_gap_se(0.90, 0.90, 1000, 0.85) - 0.01) < 1e-9
    assert abs(paired_gap_se(0.99, 0.99, 1000, 0.98) - 0.004472135955) < 1e-9


def test_paired_gap_se_rejects_impossible_agreement() -> None:
    for bad in (0.10, 0.99):
        try:
            paired_gap_se(0.90, 0.90, 1000, bad)
        except ValueError:
            continue
        raise AssertionError(f"agreement {bad} must be rejected for accuracies (0.9, 0.9)")


def test_paired_test_reports_significance() -> None:
    rng = np.random.default_rng(2)
    a = rng.random(500) < 0.90
    b = a.copy()
    result = paired_test(a, b)
    assert result.b == 0 and result.c == 0 and result.p_value == 1.0
    b = b.copy()
    b[:60] = ~b[:60]
    assert paired_test(a, b).significant


def test_selection_rules_differ_from_argmax_when_they_should() -> None:
    plateau = np.array([0.905, 0.912, 0.918, 0.914, 0.921, 0.917, 0.923, 0.919,
                        0.925, 0.922, 0.927, 0.924, 0.929, 0.926, 0.930,
                        0.928, 0.931, 0.930])
    best = int(np.argmax(plateau))
    assert select_argmax(plateau, 250) == best
    small, large = select_one_se(plateau, 250), select_one_se(plateau, 5000)
    assert small < large < best


def test_replay_optimism_is_positive_and_regret_is_non_positive() -> None:
    """Optimism is the winner's curse; regret against a max is non-positive.

    Both were previously conflated under the name `bias`, which made the bias
    figure show all-negative curves implying the opposite of the claim.
    """
    pool = _corr(10, 600, list(np.linspace(0.60, 0.95, 10)), seed=3)
    test = _corr(10, 800, list(np.linspace(0.60, 0.95, 10)), seed=4)
    result = replay_select(pool, test, "argmax", 100, 400, seed=0)
    assert result.optimism > 0, "validation accuracy of the chosen checkpoint must overstate its test accuracy"
    assert result.regret <= 0, "selection can only pick from the candidate set, so regret is non-positive"
    assert result.oracle_test == float(test.mean(axis=1).max())
    assert result.test_sampling_sd > 0


def test_replay_is_deterministic_and_rules_share_a_stream() -> None:
    pool = _corr(8, 400, list(np.linspace(0.6, 0.9, 8)), seed=5)
    test = _corr(8, 400, list(np.linspace(0.6, 0.9, 8)), seed=6)
    first = replay_select(pool, test, "argmax", 100, 200, seed=9)
    assert replay_select(pool, test, "argmax", 100, 200, seed=9).mean_selected == first.mean_selected
    rules = replay_all_rules(pool, test, 100, 150, seed=0)
    assert {r.rule for r in rules} == {"argmax", "one_se"}
    assert all(r.optimism > 0 for r in rules)


def test_gap_dispersion_matches_closed_form() -> None:
    """The like-for-like test: measured validation-gap SD vs sqrt(2p(1-p)/n)."""
    accs = list(np.linspace(0.80, 0.92, 6))
    pool = _corr(6, 20000, accs, seed=12)
    n_val = 500
    measured = validation_gap_dispersion(pool, n_val, n_draws=60, seed=1)
    theory = validation_gap_se(measured["pool_acc"], n_val)
    ratio = measured["observed_sd"] / theory
    assert 0.9 < ratio < 1.1, f"measured/theory ratio {ratio:.3f} outside 10% tolerance"


def test_stack_correctness_rejects_mismatched_pools() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, n_pool in (("a", 10), ("b", 12)):
            directory = root / name
            directory.mkdir()
            np.save(directory / "pool_correctness.npy", _corr(3, n_pool, [0.5, 0.6, 0.7]))
            np.save(directory / "test_correctness.npy", _corr(3, n_pool, [0.5, 0.6, 0.7]))
        stacked = stack_correctness([root / "a", root / "a"])
        assert stacked.shape == (6, 10)
        try:
            stack_correctness([root / "a", root / "b"])
        except ValueError:
            return
        raise AssertionError("mismatched pool sizes must be rejected")


def test_dense_average_excludes_integer_buffers() -> None:
    import tempfile
    from pathlib import Path

    model = build_model("simplecnn", 10, 1)
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.pt", Path(tmp) / "b.pt"
        torch.save(model.state_dict(), a)
        torch.save(model.state_dict(), b)
        averaged = average_state_dicts([a, b])
        for key, value in averaged.items():
            assert value.dtype == model.state_dict()[key].dtype
        assert torch.allclose(averaged["fc.weight"], model.fc.weight, atol=1e-6)


def test_top_q_selects_lowest_validation_loss() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "epochs").mkdir()
        losses = np.array([1.0, 0.5, 0.9, 0.2])
        paths = top_q_epoch_paths(root, losses, 2)
        assert [p.name for p in paths] == ["epoch_004.pt", "epoch_002.pt"]


def test_floor_stop_detects_unresolvable_tail() -> None:
    """Stops once no later epoch could beat the best so far by more than 1 SE."""
    curve = [0.60, 0.70, 0.905, 0.906, 0.907, 0.9065]
    stop = floor_stop_epoch(curve, 5000)
    assert stop is not None, "a curve that climbs then flattens must trigger the stop"
    best_so_far = max(curve[:stop + 1])
    best_later = max(curve[stop + 1:])
    assert best_later <= best_so_far + standard_error(best_so_far, 5000)
    rising = [0.10 * i for i in range(1, 11)]
    assert floor_stop_epoch(rising, 5000) is None


def test_replay_rejects_bad_sizes() -> None:
    pool = np.zeros((4, 50), dtype=bool)
    try:
        replay_select(pool, pool, "argmax", 60, 10, seed=0)
    except ValueError as exc:
        assert "must be smaller" in str(exc)
    else:
        raise AssertionError("expected ValueError for n_val >= pool size")


def test_fake_images_are_deterministic() -> None:
    a = FakeImages(10, 10, 28, eval_transform(28, "mnist"), seed=3)
    b = FakeImages(10, 10, 28, eval_transform(28, "mnist"), seed=3)
    assert torch.allclose(a[0][0], b[0][0])