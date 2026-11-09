"""The CycleCritic critics and gate, on synthetic trunks.

Everything here runs without alphafold3 or any weights: the critics read a
distogram, and nothing about them needs a real one.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from colabfold.alphafold3.cyclecritic import (
    GateSettings,
    CycleGate,
    bin_tops,
    interface_distogram_entropy,
    interface_distogram_nll,
    pseudo_beta,
    reference_distances,
    role_ids,
    threshold_at,
)

DISTOGRAM = SimpleNamespace(first_break=2.3125, last_break=21.6875, num_bins=64)


def one_hot_logits(distances, tops, confidence=20.0):
    """A trunk that is certain every distance is exactly what it is."""
    index = (np.asarray(distances)[..., None] > tops[:-1]).sum(-1)
    logits = np.zeros(index.shape + (len(tops),))
    np.put_along_axis(logits, index[..., None], confidence, axis=-1)
    return logits


def two_chain_fold(num_each=4, separation=6.0):
    """Two chains in contact: tokens 0..n-1 target, n..2n-1 binder."""
    target = np.stack([np.arange(num_each) * 3.8, np.zeros(num_each), np.zeros(num_each)], -1)
    binder = target + np.array([0.0, separation, 0.0])
    coords = np.concatenate([target, binder])
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((delta ** 2).sum(-1))
    asym = np.array([1] * num_each + [2] * num_each)
    return distances, asym, np.ones(2 * num_each, dtype=bool)


def test_pseudo_beta_sits_a_bond_length_off_the_alpha_carbon():
    n = np.array([-0.525, 1.363, 0.0])
    ca = np.array([0.0, 0.0, 0.0])
    c = np.array([1.526, 0.0, 0.0])
    cb = pseudo_beta(n, ca, c)
    assert 1.4 < np.linalg.norm(cb - ca) < 1.7
    assert np.dot(np.cross(ca - n, c - ca), cb - ca) < 0


def test_reference_distances_reads_a_two_chain_pdb(tmp_path):
    lines = []
    serial = 1
    for chain, offset in (("A", 0.0), ("B", 8.0)):
        for residue in (1, 2):
            for name, xyz in (("N", (-0.5, 1.4, 0.0)), ("CA", (0.0, 0.0, 0.0)),
                              ("C", (1.5, 0.0, 0.0))):
                x, y, z = xyz[0] + 3.8 * residue, xyz[1] + offset, xyz[2]
                lines.append(f"ATOM  {serial:5d}  {name:<3s} ALA {chain}{residue:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
                serial += 1
    lines.append(f"ATOM  {serial:5d}  CA  ALA B   3     10.000   8.000   0.000  1.00  0.00")
    (tmp_path / "design.pdb").write_text("\n".join(lines) + "\nEND\n")

    asym = np.array([1, 1, 2, 2, 2, 2])
    residue_index = np.array([1, 2, 1, 2, 3, 4])
    distances, resolved = reference_distances(
        tmp_path / "design.pdb", asym, residue_index,
        token_mask=np.ones(6, dtype=bool))
    assert resolved.tolist() == [True, True, True, True, True, False]
    assert distances.shape == (6, 6)
    assert 7.0 < distances[0, 2] < 9.0
    assert np.allclose(distances, distances.T)


def test_interface_nll_is_zero_when_the_trunk_already_believes_the_design():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, resolved = two_chain_fold()
    logits = one_hot_logits(distances, tops)
    nll = interface_distogram_nll(logits, distances, resolved, role_ids(asym), tops)
    assert nll == pytest.approx(0.0, abs=1e-5)


def test_interface_nll_is_maximal_for_a_trunk_that_knows_nothing():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, resolved = two_chain_fold()
    uniform = np.zeros(distances.shape + (len(tops),))
    nll = interface_distogram_nll(uniform, distances, resolved, role_ids(asym), tops)
    assert nll == pytest.approx(math.log(len(tops)), rel=1e-5)


def test_interface_nll_ignores_everything_but_the_cross_role_interface():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, resolved = two_chain_fold()
    logits = one_hot_logits(distances, tops)
    roles = role_ids(asym)
    before = interface_distogram_nll(logits, distances, resolved, roles, tops)

    spoiled = np.array(logits)
    spoiled[0, 1] = spoiled[1, 0] = 0.0
    far = np.argmax(np.where(roles[:, None] != roles[None, :], distances, -1))
    i, j = np.unravel_index(far, distances.shape)
    assert distances[i, j] > 12.0
    spoiled[i, j] = spoiled[j, i] = 0.0
    assert interface_distogram_nll(spoiled, distances, resolved, roles, tops) == pytest.approx(before)


def test_interface_nll_abstains_when_there_is_no_interface():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, resolved = two_chain_fold(separation=40.0)
    logits = one_hot_logits(distances, tops)
    assert math.isnan(interface_distogram_nll(logits, distances, resolved,
                                              role_ids(asym), tops))


def test_interface_nll_drops_pairs_the_design_does_not_cover():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, resolved = two_chain_fold()
    logits = np.array(one_hot_logits(distances, tops))
    resolved = resolved.copy()
    resolved[0] = False
    logits[0, :] = logits[:, 0] = 0.0
    assert interface_distogram_nll(logits, distances, resolved,
                                   role_ids(asym), tops) == pytest.approx(0.0, abs=1e-5)


def write_design(path, distances_of=two_chain_fold):
    """A PDB of the two-chain fold, so a gate can read it as its design."""
    _, asym, _ = distances_of()
    lines, serial = [], 1
    for token, chain in enumerate(asym):
        letter = "AB"[chain - 1]
        residue = int((asym[:token] == chain).sum()) + 1
        x = 3.8 * (residue - 1)
        y = 6.0 * (chain - 1)
        for name, dx, dy in (("N", -0.5, 1.4), ("CA", 0.0, 0.0), ("C", 1.5, 0.0)):
            lines.append(f"ATOM  {serial:5d}  {name:<3s} ALA {letter}{residue:4d}    "
                         f"{x + dx:8.3f}{y + dy:8.3f}{0.0:8.3f}  1.00  0.00")
            serial += 1
    Path(path).write_text("\n".join(lines) + "\nEND\n")
    residue_index = np.array([int((asym[:t] == c).sum()) + 1 for t, c in enumerate(asym)])
    return asym, residue_index


def test_a_gate_caches_the_reference_bookkeeping_without_changing_the_answer(tmp_path):
    design = tmp_path / "design.pdb"
    asym, residue_index = write_design(design)
    gate = CycleGate(GateSettings(design=design),
                     distogram_config=DISTOGRAM, asym_id=asym,
                     token_mask=np.ones(len(asym), dtype=bool))
    gate.load_reference(residue_index)
    distances, resolved = gate.reference
    assert resolved.all()
    assert gate.has_structure_critic

    rng = np.random.default_rng(3)
    logits = rng.normal(size=distances.shape + (len(gate.tops),))
    uncached = interface_distogram_nll(logits, distances, resolved, gate.roles, gate.tops)
    cached = interface_distogram_nll(logits, distances, resolved, gate.roles, gate.tops,
                                     bins=gate._bins, mask=gate._interface)
    assert cached == pytest.approx(uncached)
    gate.after_pass(0, logits)
    assert gate.history[0].scores["interface_nll"] == pytest.approx(uncached)


def test_entropy_proxy_is_high_for_a_sharp_interface_and_low_for_a_vague_one():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, resolved = two_chain_fold()
    roles, mask = role_ids(asym), np.ones(len(asym), dtype=bool)
    sharp = interface_distogram_entropy(one_hot_logits(distances, tops), roles, mask, tops)

    vague = np.full(distances.shape + (len(tops),), -1e9)
    contact = tops <= 8.001
    vague[..., contact] = 0.0
    blurred = interface_distogram_entropy(vague, roles, mask, tops)
    assert 0.9 < sharp <= 1.0
    assert 0.0 < blurred < sharp


def test_entropy_proxy_abstains_when_the_trunk_sees_no_interface_at_all():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, _ = two_chain_fold()
    logits = np.full(distances.shape + (len(tops),), -1e9)
    logits[..., -1] = 0.0
    assert math.isnan(interface_distogram_entropy(
        logits, role_ids(asym), np.ones(len(asym), dtype=bool), tops))


def test_entropy_proxy_ignores_padding_tokens():
    tops = bin_tops(**vars(DISTOGRAM))
    distances, asym, _ = two_chain_fold()
    roles = role_ids(asym)
    logits = one_hot_logits(distances, tops)
    mask = np.ones(len(asym), dtype=bool)
    full = interface_distogram_entropy(logits, roles, mask, tops)

    padded_logits = np.zeros((len(asym) + 2,) * 2 + (len(tops),))
    padded_logits[:len(asym), :len(asym)] = logits
    padded_roles = np.concatenate([roles, [0, 0]])
    padded_mask = np.concatenate([mask, [False, False]])
    assert interface_distogram_entropy(padded_logits, padded_roles, padded_mask,
                                       tops) == pytest.approx(full)


def test_thresholds_parse_and_hold_their_last_value():
    assert GateSettings.parse_thresholds("0.3, 0.2,0.1") == (0.3, 0.2, 0.1)
    assert GateSettings.parse_thresholds("off") is None
    assert GateSettings.parse_thresholds(None) is None
    schedule = (0.3, 0.2, 0.1)
    assert threshold_at(schedule, 0) == 0.3
    assert threshold_at(schedule, 7) == 0.1
    assert threshold_at(None, 0) is None


def make_gate(**kwargs):
    distances, asym, _ = two_chain_fold()
    settings = GateSettings(**kwargs)
    gate = CycleGate(settings, distogram_config=DISTOGRAM, asym_id=asym,
                     token_mask=np.ones(len(asym), dtype=bool))
    return gate, distances, asym


def test_the_gate_waits_for_min_cycles_before_it_kills_anything():
    gate, distances, _ = make_gate(min_cycles=3, confidence_thresholds=(0.99,))
    tops = gate.tops
    flat = np.zeros(distances.shape + (len(tops),))
    assert gate.after_pass(0, flat) is None
    assert gate.after_pass(1, flat) is None
    assert gate.after_pass(2, flat) == "confidence"
    report = gate.report()
    assert report["killed"] and report["exit_cycle"] == 2 and report["trunk_passes"] == 3


def test_the_gate_keeps_a_refold_its_critics_are_happy_with():
    gate, distances, _ = make_gate(confidence_thresholds=(0.5,))
    logits = one_hot_logits(distances, gate.tops)
    assert gate.after_pass(0, logits) is None
    assert gate.report()["killed"] is False


def test_the_structure_critic_kills_a_refold_that_disagrees_with_the_design(tmp_path):
    distances, asym, resolved = two_chain_fold()
    settings = GateSettings(structure_thresholds=(1.0,),
                            design=tmp_path / "unused.pdb")
    gate = CycleGate(settings, distogram_config=DISTOGRAM, asym_id=asym,
                     token_mask=np.ones(len(asym), dtype=bool))
    gate.reference = (distances, resolved)
    assert gate.has_structure_critic

    agrees = one_hot_logits(distances, gate.tops)
    assert gate.after_pass(0, agrees) is None
    assert gate.after_pass(1, np.zeros(agrees.shape)) == "structure"
    assert gate.report()["killed_by"] == "structure"


def test_a_gate_without_thresholds_scores_but_never_kills():
    gate, distances, _ = make_gate(trace=True)
    flat = np.zeros(distances.shape + (len(gate.tops),))
    for cycle in range(4):
        assert gate.after_pass(cycle, flat) is None
    report = gate.report()
    assert report["killed"] is False
    assert len(report["per_cycle"]) == 4
    assert "entropy_proxy" in report["per_cycle"][0]


def test_a_gate_with_no_design_runs_the_confidence_critic_only():
    gate, distances, _ = make_gate()
    tops = bin_tops(**vars(DISTOGRAM))
    gate.after_pass(0, one_hot_logits(distances, tops))
    assert not gate.has_structure_critic
    assert gate.report()["critics"] == ["entropy_proxy"]


def test_naming_the_binder_chain_makes_the_interface_binder_to_target():
    asym = np.array([1, 1, 2, 2, 3, 3])
    settings = GateSettings(binder_chain="C")
    gate = CycleGate(settings, distogram_config=DISTOGRAM, asym_id=asym,
                     token_mask=np.ones(6, dtype=bool))
    assert gate.roles.tolist() == [0, 0, 0, 0, 1, 1]


def test_an_unknown_binder_chain_is_an_error():
    with pytest.raises(ValueError, match="not a chain of this fold"):
        CycleGate(GateSettings(binder_chain="Z"),
                  distogram_config=DISTOGRAM, asym_id=np.array([1, 1, 2, 2]),
                  token_mask=np.ones(4, dtype=bool))


class StagedRunner:
    """``ModelRunner.run_inference_gated`` over stages that only count calls.

    The stages themselves are alphafold3's; what is tested here is the driver
    around them -- that it recycles the right number of times, asks the gate after
    every pass, computes the distogram only for a gate that reads one, and stops
    before scoring when the gate fires.
    """

    def __init__(self, passes=4):
        from colabfold.alphafold3.models import ModelRunner

        self.calls = {"trunk": 0, "distogram": 0, "score": 0}
        stages = self._stages()

        class Runner(ModelRunner):
            """The real driver; everything it drives is a stub."""

            model_params = {}
            num_trunk_passes = passes
            _stages = stages

            def _to_device(self, example):
                return example

            def _to_host(self, result):
                return dict(result)

            @staticmethod
            def _preinit_tokamax_context():
                pass

        self.runner = Runner(
            SimpleNamespace(heads=SimpleNamespace(distogram=DISTOGRAM)), model_dir=".")

    def _stages(self):
        tops = bin_tops(**vars(DISTOGRAM))
        distances, asym, _ = two_chain_fold()
        logits = one_hot_logits(distances, tops)

        def trunk(params, rng, batch, carry, key):
            self.calls["trunk"] += 1
            return {"pair": np.zeros((len(asym), len(asym), 4))}, key, None

        def distogram(params, rng, batch, carry, key):
            self.calls["distogram"] += 1
            return {"distogram": {"distogram": logits}}

        def score(params, rng, batch, carry, key):
            self.calls["score"] += 1
            return {"diffusion_samples": {}}

        return {"embed": lambda *a: ({}, "key"), "trunk": trunk,
                "distogram": distogram, "score": score}


def test_a_surviving_refold_runs_every_pass_and_then_the_heads():
    staged = StagedRunner(passes=4)
    gate, _, _ = make_gate(confidence_thresholds=(0.1,))
    result = staged.runner.run_inference_gated({}, "rng", gate)
    assert result is not None
    assert staged.calls == {"trunk": 4, "distogram": 4, "score": 1}
    assert gate.report()["trunk_passes"] == 4


def test_a_killed_refold_never_reaches_diffusion_or_the_heads():
    staged = StagedRunner(passes=10)
    gate, _, _ = make_gate(confidence_thresholds=(1.01,))
    assert staged.runner.run_inference_gated({}, "rng", gate) is None
    assert staged.calls == {"trunk": 1, "distogram": 1, "score": 0}
    assert gate.report()["exit_cycle"] == 0


def options(**backend_opts):
    from colabfold.backend import RunOptions

    return RunOptions(model_type="protenix2", backend_opts=backend_opts)


def test_gating_is_off_unless_it_is_asked_for():
    from colabfold.alphafold3.cyclecritic import settings_from_options

    assert settings_from_options(options()) is None


def test_options_become_settings():
    from colabfold.alphafold3.cyclecritic import settings_from_options

    settings = settings_from_options(options(
        cyclecritic=True, cyclecritic_design="design.cif",
        cyclecritic_structure_thresholds="4.0,3.5", cyclecritic_min_cycles=2,
        cyclecritic_binder_chain="B"))
    assert settings.structure_thresholds == (4.0, 3.5)
    assert settings.min_cycles == 2
    assert settings.binder_chain == "B"
    assert str(settings.design) == "design.cif"
    assert settings.confidence_thresholds is None


def test_the_structure_critic_refuses_to_gate_without_a_design():
    from colabfold.alphafold3.cyclecritic import settings_from_options

    with pytest.raises(ValueError, match="needs the designed backbone"):
        settings_from_options(options(cyclecritic=True,
                                      cyclecritic_structure_thresholds="4.0"))


def test_an_initial_guess_is_the_design_the_structure_critic_scores_against():
    from colabfold.alphafold3.cyclecritic import settings_from_options
    from colabfold.backend import RunOptions

    opts = RunOptions(model_type="protenix2", initial_guess="design.cif",
                      backend_opts={"cyclecritic": True})
    assert str(settings_from_options(opts).design) == "design.cif"
