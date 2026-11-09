"""
Run an alphafold3-open model and write results in ColabFold's layout.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def per_residue_plddt(structure) -> List[float]:
    """Mean atom b-factor per residue, in chain order."""
    b = np.asarray(structure.atom_b_factor, dtype=float)
    chain = np.asarray(structure.chain_id)
    res = np.asarray(structure.res_id)
    if len(b) == 0:
        return []
    keys = np.asarray([f"{c}\t{r}" for c, r in zip(chain, res)])
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    return [float(v) for v in np.add.reduceat(b, starts) / np.diff(np.append(starts, len(b)))]


def scores_of(inference_result, ranking_score: float) -> Dict[str, Any]:
    from alphafold3.model import confidence_types

    full = confidence_types.StructureConfidenceFull.from_inference_result(inference_result)
    meta = inference_result.metadata
    pae = np.asarray(full.pae, dtype=float)
    scores = {
        "plddt": np.around(per_residue_plddt(inference_result.predicted_structure), 2).tolist(),
        "pae": np.around(pae, 2).tolist(),
        "max_pae": float(pae.max()),
        "ranking_score": float(ranking_score),
        "token_chain_ids": list(full.token_chain_ids),
    }
    for key, name in (("predicted_tm_score", "ptm"),
                      ("interface_predicted_tm_score", "iptm"),
                      ("has_clash", "has_clash"),
                      ("fraction_disordered", "fraction_disordered")):
        value = meta.get(key)
        if value is not None and np.isfinite(float(value)):
            scores[name] = float(value)
    return scores


def _print_line(tag: str, scores: Dict[str, Any], took: float) -> str:
    from colabfold.backend import metrics_line

    summary = {k: scores[k] for k in ("ptm", "iptm", "ranking_score") if k in scores}
    if scores["plddt"]:
        summary["mean_plddt"] = float(np.mean(scores["plddt"]))
    return f"{tag} took {took:.1f}s" + metrics_line(summary)


def rank_key(rank_by: str, scores: Dict[str, Any]) -> float:
    """The metric to rank on. ``auto`` is alphafold3's own ranking score."""
    if rank_by in ("auto", "ranking_score"):
        return float(scores["ranking_score"])
    if rank_by == "plddt":
        return float(np.mean(scores["plddt"])) if scores["plddt"] else float("nan")
    if rank_by == "multimer":
        rank_by = "iptm" if "iptm" in scores else "ptm"
    return float(scores.get(rank_by, scores["ranking_score"]))


def build_gate(gate_settings, model_runner, fold_input, example):
    """One :class:`colabfold.alphafold3.cyclecritic.CycleGate` for one refold."""
    from colabfold.alphafold3.cyclecritic import CycleGate

    gate = CycleGate(
        gate_settings,
        distogram_config=model_runner.distogram_config,
        asym_id=example["asym_id"],
        token_mask=example["seq_mask"],
        chain_letters=[chain.id for chain in fold_input.chains],
    )
    gate.load_reference(example["residue_index"])
    return gate


def _gate_summary(reports: Dict[int, Dict[str, Any]], full_passes: int) -> str:
    killed = [r for r in reports.values() if r["killed"]]
    passes = [r["trunk_passes"] for r in reports.values()]
    mean = sum(passes) / len(passes)
    by_critic = {}
    for report in killed:
        by_critic[report["killed_by"]] = by_critic.get(report["killed_by"], 0) + 1
    tail = "".join(f", {n} by the {critic} critic" for critic, n in sorted(by_critic.items()))
    return (f"CycleCritic stopped {len(killed)} of {len(reports)} refold(s){tail}; "
            f"mean trunk depth {mean:.2f} of {full_passes} pass(es) "
            f"({100 * (1 - mean / full_passes):.1f}% fewer)")


def predict_structure(
    prefix: str,
    result_dir: Path,
    fold_input,
    model_runner,
    model_type: str,
    featurised_examples: List[Any],
    save_all: bool = False,
    rank_by: str = "auto",
    stop_at_score: float = 100.0,
    prediction_callback=None,
    gate_settings=None,
) -> Dict[str, Any]:
    import jax
    from alphafold3.model import post_processing

    ranked, best = [], -float("inf")
    gate_reports: Dict[int, Dict[str, Any]] = {}
    extra_files: List[Path] = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        if best >= stop_at_score:
            logger.info(f"stopping early, {rank_by} reached {best:.3f}")
            break
        start = time.time()
        if gate_settings is not None:
            gate = build_gate(gate_settings, model_runner, fold_input, example)
            result = model_runner.run_inference_gated(
                example, jax.random.PRNGKey(seed), gate)
            gate_reports[seed] = gate.report()
            if result is None:
                report = gate_reports[seed]
                logger.info(f"{model_type}_seed_{seed:03d}: the {report['killed_by']} "
                            f"CycleCritic stopped this refold after "
                            f"{report['trunk_passes']} trunk pass(es) "
                            f"({time.time() - start:.1f}s)")
                continue
        else:
            result = model_runner.run_inference(example, jax.random.PRNGKey(seed))
        inference_results = model_runner.extract_inference_results(
            batch=example, result=result, target_name=fold_input.name
        )
        took = time.time() - start
        for sample, inference_result in enumerate(inference_results):
            processed = post_processing.post_process_inference_result(inference_result)
            tag = f"{model_type}_seed_{seed:03d}_sample_{sample}"
            scores = scores_of(inference_result, processed.ranking_score)
            num_tokens = int(np.sum(example["seq_mask"]))
            logits = model_runner.extract_logits(result, sample, num_tokens) if save_all else None
            ranked.append((rank_key(rank_by, scores), tag, processed, scores, took,
                           model_runner.extract_embeddings(result, num_tokens), logits))
            best = max(best, ranked[-1][0])
            if prediction_callback is not None:
                prediction_callback(inference_result.predicted_structure, None,
                                    scores, example, (tag, False))

    if gate_reports:
        logger.info(_gate_summary(gate_reports, model_runner.num_trunk_passes))
        gate_file = result_dir.joinpath(f"{prefix}_cyclecritic.json")
        gate_file.write_text(json.dumps(
            {"full_trunk_passes": model_runner.num_trunk_passes,
             "seeds": {str(seed): report for seed, report in gate_reports.items()}},
            indent=2))
        extra_files.append(gate_file)

    logger.info(f"reranking models by '{rank_by}' metric")
    ranked.sort(key=lambda row: row[0], reverse=True)

    rank, metric, result_files = [], [], list(extra_files)
    for n, (_, tag, processed, scores, took, embeddings, logits) in enumerate(ranked):
        new_tag = f"rank_{(n + 1):03d}_{tag}"
        rank.append(new_tag)
        metric.append(scores)
        logger.info(_print_line(new_tag, scores, took))
        cif = result_dir.joinpath(f"{prefix}_{new_tag}.cif")
        cif.write_bytes(processed.cif)
        result_files.append(cif)
        scores_file = result_dir.joinpath(f"{prefix}_scores_{new_tag}.json")
        scores_file.write_text(json.dumps(scores))
        result_files.append(scores_file)
        if save_all:
            full = result_dir.joinpath(f"{prefix}_confidences_{new_tag}.json")
            full.write_bytes(processed.structure_full_data_json)
            result_files.append(full)
        if logits:
            npz = result_dir.joinpath(f"{prefix}_logits_{new_tag}.npz")
            np.savez_compressed(npz, **logits)
            result_files.append(npz)
        if embeddings:
            npz = result_dir.joinpath(f"{prefix}_representations_{new_tag}.npz")
            np.savez_compressed(npz, **embeddings)
            result_files.append(npz)
    return {"rank": rank, "metric": metric, "result_files": result_files}
