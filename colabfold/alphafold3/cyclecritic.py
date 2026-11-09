"""CycleCritic: refold quality predicted from the trunk, at every recycle.

Vaitl, Tanasijevic, Jamasb, Mille-Fragoso and Kessel, "Recycle only when you have
to!" (arXiv:2610.40665) observe that an AlphaFold3-like model's confidence and
self-consistency metrics are largely determined by the trunk, and can be read off
it several recycles before the schedule ends. A screening run can therefore stop
a hopeless refold between recycles, before diffusion and the confidence head ever
run.

Two critics read the trunk's distogram after every recycle. Both are the
parameter-free form given in the paper's section 4.1:

* the STRUCTURE critic scores the trunk against the designed backbone the refold
  is meant to reproduce: the mean negative log-likelihood of the design's
  interface distances under the distogram (their Algorithm 1). It rises with
  disagreement, and needs a reference structure.
* the CONFIDENCE critic scores how constrained the trunk's interface belief is,
  as a stand-in for ipTM. It rises with confidence, and needs no reference, so it
  is the one critic available for an ordinary fold.

The gate is KILL-ONLY: a refold that survives every recycle is folded and scored
exactly as it would have been without a gate, so a gated run can only cost
recycles, never change a structure it keeps.
"""
import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INTERFACE_CUTOFF = 12.0

_CB_FROM_BACKBONE = (-0.58273431, 0.56802827, -0.54067466)


def pseudo_beta(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """A virtual Cb from one residue's backbone atoms, in trRosetta's geometry.

    The paper's structure critic reads "Cb constructed from N, CA and C", so the
    reference is backbone-only and independent of side chains (appendix D.2).
    """
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    wa, wb, wc = _CB_FROM_BACKBONE
    return wa * a + wb * b + wc * cc + ca


def _residue_representatives(path: Path) -> Dict[Tuple[str, int], np.ndarray]:
    """``(chain, residue number) -> representative atom`` for one structure file.

    Cb for a residue with a complete backbone, the real Cb for a residue whose
    backbone is incomplete but which has one, CA otherwise, and the first atom for
    anything that is not an amino acid (a nucleotide's or a ligand's position is
    still a position the trunk has an opinion about). A residue's key is its
    author numbering, which is what ColabFold's own output and a design pipeline's
    both carry.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(path)
    parser = (MMCIFParser(QUIET=True) if path.suffix.lower() in (".cif", ".mmcif")
              else PDBParser(QUIET=True))
    structure = parser.get_structure("design", str(path))
    out: Dict[Tuple[str, int], np.ndarray] = {}
    for chain in next(structure.get_models()):
        for residue in chain:
            atoms = {atom.get_name(): np.asarray(atom.get_coord(), dtype=float)
                     for atom in residue}
            if all(name in atoms for name in ("N", "CA", "C")):
                representative = pseudo_beta(atoms["N"], atoms["CA"], atoms["C"])
            elif "CB" in atoms:
                representative = atoms["CB"]
            elif "CA" in atoms:
                representative = atoms["CA"]
            elif atoms:
                representative = next(iter(atoms.values()))
            else:
                continue
            out[(chain.id, int(residue.get_id()[1]))] = representative
    return out


def reference_distances(path: Path, asym_id: Sequence[int], residue_index: Sequence[int],
                        token_mask: Optional[Sequence[float]] = None,
                        chain_ids: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The designed backbone's pairwise distances, on the fold's own tokens.

    Returns ``(distances, resolved)``: an (N, N) matrix in Angstrom and the mask
    of tokens the reference actually has an atom for. A token the design does not
    cover -- a residue that is missing, or any token past the real sequence in a
    padded batch -- is left unresolved rather than guessed at, and every critic
    here drops the pairs it appears in.

    Chains are matched by name. With ``chain_ids`` unset the fold's chains are
    assumed to be A, B, C, ... in order, which is how alphafold3 builds its 1-based
    asym ids and how ColabFold names the chains it hands over.
    """
    from colabfold.alphafold3.input import chain_id

    asym = np.asarray(asym_id).astype(int)
    res = np.asarray(residue_index).astype(int)
    if chain_ids is None:
        chain_ids = [chain_id(int(a) - 1) for a in np.unique(asym[asym > 0])]
    letters = {int(a): letter for a, letter in zip(np.unique(asym[asym > 0]), chain_ids)}
    representatives = _residue_representatives(path)

    coords = np.zeros((len(asym), 3), dtype=float)
    resolved = np.zeros(len(asym), dtype=bool)
    for token, (a, r) in enumerate(zip(asym, res)):
        if token_mask is not None and not token_mask[token]:
            continue
        xyz = representatives.get((letters.get(int(a), ""), int(r)))
        if xyz is None:
            continue
        coords[token] = xyz
        resolved[token] = True
    real = (np.ones(len(asym), dtype=bool) if token_mask is None
            else np.asarray(token_mask).astype(bool))
    missing = int((real & ~resolved).sum())
    if missing:
        logger.warning(f"the reference design covers {int(resolved.sum())} of "
                       f"{int(real.sum())} tokens; the {missing} it does not are "
                       f"dropped from the structure critic")
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((delta ** 2).sum(-1)), resolved


def bin_tops(first_break: float, last_break: float, num_bins: int) -> np.ndarray:
    """The distogram's upper bin edges, exactly as ``DistogramHead`` builds them."""
    breaks = np.linspace(first_break, last_break, num_bins - 1)
    return np.append(breaks, breaks[-1] + (breaks[-1] - breaks[-2]))


def bin_index(distances, tops) -> np.ndarray:
    """Which distogram bin each distance falls in."""
    index = (np.asarray(distances)[..., None] > np.asarray(tops[:-1])).sum(-1)
    return np.minimum(index, len(tops) - 1)


def role_ids(asym_id: Sequence[int], binder_asym: Optional[int] = None) -> np.ndarray:
    """Per-token role: 1 for the binder, 0 for the target.

    The paper scores a binder against a target, so the pairs that matter are the
    cross-role ones. ColabFold folds complexes that have no designated binder, so
    with ``binder_asym`` unset every chain is its own role and "cross-role" means
    "cross-chain", which is the same thing for the two-chain case the paper runs.
    """
    asym = np.asarray(asym_id).astype(int)
    if binder_asym is None:
        return asym
    return (asym == int(binder_asym)).astype(int)


def cross_role(roles: Sequence[int]) -> np.ndarray:
    roles = np.asarray(roles)
    return roles[:, None] != roles[None, :]


def interface_mask(ref_dist, resolved, roles, cutoff: float = INTERFACE_CUTOFF):
    """Algorithm 1's ``M``: cross-role, distinct, in contact, both tokens resolved."""
    import jax.numpy as jnp

    resolved = jnp.asarray(resolved).astype(bool)
    both = resolved[:, None] & resolved[None, :]
    distinct = ~jnp.eye(len(resolved), dtype=bool)
    return (jnp.asarray(cross_role(roles)) & distinct & both
            & (jnp.asarray(ref_dist) < cutoff))


def interface_distogram_nll(logits, ref_dist, resolved, roles, tops,
                            cutoff: float = INTERFACE_CUTOFF, *, bins=None,
                            mask=None) -> float:
    """Algorithm 1: the design's interface distances' surprise under the trunk.

    The mean negative log-likelihood, under the trunk's own distogram at this
    recycle, of the bin each interface distance of the designed backbone falls in.
    Low is good: the trunk already believes the pose the design asks for. Returns
    NaN when the design has no interface, where the paper's gate abstains.

    ``bins`` and ``mask`` depend on the reference alone, so a gate that scores ten
    recycles of the same design computes them once and hands them back.
    """
    import jax
    import jax.numpy as jnp

    log_p = jax.nn.log_softmax(jnp.asarray(logits), axis=-1)
    if bins is None:
        bins = bin_index(ref_dist, tops)
    if mask is None:
        mask = interface_mask(ref_dist, resolved, roles, cutoff)
    surprise = -jnp.take_along_axis(log_p, jnp.asarray(bins)[..., None], axis=-1)[..., 0]
    count = mask.sum()
    return float(jnp.where(count > 0, (surprise * mask).sum() / jnp.maximum(count, 1),
                           jnp.nan))


def interface_distogram_entropy(logits, roles, token_mask, tops,
                                contact_threshold: float = 8.0) -> float:
    """The ipTM proxy: how constrained the trunk's interface belief is.

    Candido et al. use the entropy of the interface distogram as a training-free
    stand-in for ipTM, on the argument that the structure module places residues
    conditioned on the trunk, so a sharply peaked distogram is a confident pose
    and a flat one is not (paper section 4.1, which describes the mechanism
    without reproducing the algorithm). This follows that description: the
    contact-probability-weighted mean of the distogram's entropy over cross-role
    pairs, normalised by ``log K`` and flipped, so the score rises with confidence
    and lies in [0, 1].

    Returns NaN when no cross-role pair carries any contact mass, i.e. the trunk
    believes there is no interface at all to be confident about.
    """
    import jax
    import jax.numpy as jnp

    probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    contact = (jnp.asarray(tops) <= contact_threshold + 1e-3).astype(probs.dtype)
    weight = jnp.einsum("ijk,k->ij", probs, contact)

    mask = jnp.asarray(token_mask).astype(bool)
    pairs = jnp.asarray(cross_role(roles)) & mask[:, None] & mask[None, :]
    weight = weight * pairs

    entropy = -(probs * jnp.log(jnp.clip(probs, 1e-9, None))).sum(-1)
    total = weight.sum()
    mean = (weight * entropy).sum() / jnp.maximum(total, 1e-9)
    score = 1.0 - mean / math.log(probs.shape[-1])
    return float(jnp.where(total > 1e-6, score, jnp.nan))


@dataclasses.dataclass
class GateSettings:
    """How a gated refold spends its recycles.

    Thresholds are read per recycle and the last one is held for any recycle past
    the end of the list, which is what lets one schedule serve a model run at more
    recycles than it was calibrated at.

    Note which way each threshold points: the structure critic kills a refold
    whose interface NLL is ABOVE its threshold, the confidence critic one whose
    entropy proxy is BELOW it. Their scale depends on the model and the design
    pipeline, so no default is shipped -- calibrate them on refolds whose outcome
    is known, which is what ``--cyclecritic-trace`` records.
    """
    min_cycles: int = 1
    confidence_thresholds: Optional[Tuple[float, ...]] = None
    structure_thresholds: Optional[Tuple[float, ...]] = None
    design: Optional[Path] = None
    binder_chain: Optional[str] = None
    trace: bool = False

    @staticmethod
    def parse_thresholds(value) -> Optional[Tuple[float, ...]]:
        """``"0.3,0.27,0.24"``, ``"off"`` or a number for every recycle."""
        if value is None or value == "" or value == "off":
            return None
        if isinstance(value, (list, tuple)):
            return tuple(float(x) for x in value)
        text = str(value).strip()
        return tuple(float(x) for x in text.replace(" ", "").split(","))


def threshold_at(thresholds: Optional[Sequence[float]], cycle: int) -> Optional[float]:
    if not thresholds:
        return None
    return float(thresholds[min(cycle, len(thresholds) - 1)])


@dataclasses.dataclass
class Decision:
    """What the critics said after one trunk pass."""
    cycle: int
    scores: Dict[str, float]
    killed_by: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"cycle": self.cycle}
        out.update({k: (None if v is None or not np.isfinite(v) else round(float(v), 6))
                    for k, v in self.scores.items()})
        if self.killed_by:
            out["killed_by"] = self.killed_by
        return out


class CycleGate:
    """The paper's union gate, driven one trunk pass at a time.

    ``after_pass`` is called after every recycle with that pass's distogram
    logits, and returns the name of the critic that killed the refold, or None to
    carry on. Either critic firing stops the refold after the current recycle,
    before diffusion and the confidence head -- the gates are kill-only, so a
    survivor is folded exactly as an ungated run would fold it.
    """

    def __init__(self, settings: GateSettings, *, distogram_config,
                 asym_id, token_mask, chain_letters: Optional[Sequence[str]] = None):
        self.settings = settings
        self.tops = bin_tops(distogram_config.first_break,
                             distogram_config.last_break,
                             distogram_config.num_bins)
        self.token_mask = np.asarray(token_mask).astype(bool)
        self.asym_id = np.asarray(asym_id).astype(int)
        self.chain_letters = list(chain_letters) if chain_letters else None
        binder_asym = self._binder_asym()
        self.roles = role_ids(self.asym_id, binder_asym)
        self.reference = None
        self._bins = self._interface = None
        self._design_path = Path(settings.design) if settings.design else None
        self.history: List[Decision] = []
        self.passes = 0

    def _binder_asym(self) -> Optional[int]:
        """The asym id of the chain named as the binder, if one was."""
        letter = self.settings.binder_chain
        if not letter:
            return None
        from colabfold.alphafold3.input import chain_id

        for asym in np.unique(self.asym_id[self.asym_id > 0]):
            name = (self.chain_letters[int(asym) - 1] if self.chain_letters
                    and len(self.chain_letters) >= int(asym) else chain_id(int(asym) - 1))
            if name == letter:
                return int(asym)
        raise ValueError(f"--cyclecritic-binder-chain {letter!r} is not a chain of this "
                         f"fold (it has {sorted(set(self.asym_id[self.asym_id > 0]))} chains)")

    def load_reference(self, residue_index) -> None:
        """Read the designed backbone, once the fold's residue numbering is known.

        The per-pair bookkeeping every recycle reuses -- which distogram bin each
        reference distance falls in, and which pairs are the interface -- is built
        here too, since none of it changes as the trunk recycles.
        """
        if self._design_path is None:
            return
        distances, resolved = reference_distances(
            self._design_path, self.asym_id, residue_index,
            token_mask=self.token_mask, chain_ids=self.chain_letters)
        self.reference = (distances, resolved)
        self._bins = bin_index(distances, self.tops)
        self._interface = np.asarray(interface_mask(distances, resolved, self.roles))

    @property
    def has_structure_critic(self) -> bool:
        """Whether a designed backbone was given for the structure critic to score."""
        return self.reference is not None

    def after_pass(self, cycle: int, logits) -> Optional[str]:
        self.passes = cycle + 1
        scores: Dict[str, float] = {}
        killed_by = None
        structure = None

        confidence = interface_distogram_entropy(
            logits, self.roles, self.token_mask, self.tops)
        scores["entropy_proxy"] = confidence
        if self.has_structure_critic:
            distances, resolved = self.reference
            structure = interface_distogram_nll(
                logits, distances, resolved, self.roles, self.tops,
                bins=self._bins, mask=self._interface)
            scores["interface_nll"] = structure

        if cycle + 1 >= max(1, self.settings.min_cycles):
            killed_by = self._verdict(cycle, confidence, structure)
        self.history.append(Decision(cycle, scores, killed_by))
        return killed_by

    def _verdict(self, cycle: int, confidence, structure) -> Optional[str]:
        """Either critic past its bar kills the refold.

        The entropy proxy rises with quality, so it kills from below; the NLL is a
        surprise, so it kills from above.
        """
        bar = threshold_at(self.settings.confidence_thresholds, cycle)
        if bar is not None and confidence is not None and np.isfinite(confidence):
            if confidence < bar:
                return "confidence"
        bar = threshold_at(self.settings.structure_thresholds, cycle)
        if bar is not None and structure is not None and np.isfinite(structure):
            if structure > bar:
                return "structure"
        return None

    def report(self) -> Dict[str, Any]:
        killed = next((d for d in self.history if d.killed_by), None)
        out: Dict[str, Any] = {
            "trunk_passes": self.passes,
            "killed": killed is not None,
            "killed_by": killed.killed_by if killed else None,
            "exit_cycle": killed.cycle if killed else None,
            "critics": sorted({name for d in self.history for name in d.scores}),
        }
        if self.settings.trace:
            out["per_cycle"] = [d.as_dict() for d in self.history]
        else:
            out["last_cycle"] = self.history[-1].as_dict() if self.history else None
        return out


def settings_from_options(opts) -> Optional[GateSettings]:
    """A ``GateSettings`` from ColabFold's backend options, or None if gating is off."""
    if not opts.opt("cyclecritic", False):
        return None
    design = opts.opt("cyclecritic_design") or opts.initial_guess
    settings = GateSettings(
        min_cycles=int(opts.opt("cyclecritic_min_cycles", 1) or 1),
        confidence_thresholds=GateSettings.parse_thresholds(
            opts.opt("cyclecritic_confidence_thresholds")),
        structure_thresholds=GateSettings.parse_thresholds(
            opts.opt("cyclecritic_structure_thresholds")),
        design=Path(design) if design else None,
        binder_chain=opts.opt("cyclecritic_binder_chain"),
        trace=bool(opts.opt("cyclecritic_trace", False)),
    )
    if settings.structure_thresholds is not None and settings.design is None:
        raise ValueError("--cyclecritic-structure-thresholds needs the designed "
                         "backbone the refold is scored against: pass "
                         "--cyclecritic-design")
    if settings.confidence_thresholds is None and settings.structure_thresholds is None:
        logger.warning("--cyclecritic has no thresholds, so it will score every "
                       "recycle and kill nothing. Calibrate on refolds whose outcome "
                       "you know (--cyclecritic-trace records the per-cycle scores) "
                       "and pass --cyclecritic-confidence-thresholds / "
                       "--cyclecritic-structure-thresholds.")
    return settings
