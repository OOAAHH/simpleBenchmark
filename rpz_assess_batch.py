#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ChainMappingResult:
    ok: bool
    mapping: Dict[str, str]  # model_chain -> ref_chain
    source: str  # "auto" | "explicit:<path>"
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    similarity_matrix: Optional[List[List[float]]] = None


@dataclass(frozen=True)
class ChainInfo:
    chain_id: str
    sequence: str
    residues: Tuple[Tuple[int, str], ...]  # (resSeq, iCode)


@dataclass(frozen=True)
class FilterResult:
    input_path: Path
    output_path: Path
    ok: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    chains: Dict[str, ChainInfo]
    atoms_in: int
    atoms_out: int
    residues_out: int


@dataclass(frozen=True)
class DockerConfig:
    platform: str
    tools_image: str
    gdt_image: str
    ost_image: str


@dataclass(frozen=True)
class InputsConfig:
    prefer_processed: bool
    references_dirname: str
    models_dirname: str
    processed_dirname: str
    residues_list: Path
    atoms_list: Path
    chain_mapping_dirname: str
    chain_mapping_filename: str


@dataclass(frozen=True)
class WhitelistConfig:
    backbone: Tuple[str, ...]
    heavy: Tuple[str, ...]

    @property
    def all_atoms(self) -> Tuple[str, ...]:
        return self.backbone + self.heavy


@dataclass(frozen=True)
class ParallelismConfig:
    max_workers: int


@dataclass(frozen=True)
class OutputsConfig:
    results_root: Path


@dataclass(frozen=True)
class AppConfig:
    docker: DockerConfig
    inputs: InputsConfig
    whitelist: WhitelistConfig
    parallelism: ParallelismConfig
    outputs: OutputsConfig


def _load_config(path: Path) -> AppConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = int(raw.get("version", 0))
    if version != 1:
        raise ValueError(f"Unsupported config version: {version} (expected 1)")

    def _resolve_repo_path(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (REPO_ROOT / pp)

    docker_raw = raw["docker"]
    images = docker_raw["images"]
    docker = DockerConfig(
        platform=str(docker_raw.get("platform", "linux/amd64")),
        tools_image=str(images["tools"]),
        gdt_image=str(images["gdt"]),
        ost_image=str(images["ost"]),
    )

    inputs_raw = raw["inputs"]
    inputs = InputsConfig(
        prefer_processed=bool(inputs_raw.get("prefer_processed", True)),
        references_dirname=str(inputs_raw.get("references_dirname", "references")),
        models_dirname=str(inputs_raw.get("models_dirname", "models")),
        processed_dirname=str(inputs_raw.get("processed_dirname", "processed")),
        residues_list=_resolve_repo_path(str(inputs_raw["residues_list"])),
        atoms_list=_resolve_repo_path(str(inputs_raw["atoms_list"])),
        chain_mapping_dirname=str(inputs_raw.get("chain_mapping_dirname", "chain_mappings")),
        chain_mapping_filename=str(inputs_raw.get("chain_mapping_filename", "chain_mapping.json")),
    )

    wl = raw["atoms_whitelist"]
    whitelist = WhitelistConfig(
        backbone=tuple(str(x) for x in wl["backbone"]),
        heavy=tuple(str(x) for x in wl["heavy"]),
    )

    par_raw = raw.get("parallelism", {})
    parallelism = ParallelismConfig(max_workers=int(par_raw.get("max_workers", 8)))

    out_raw = raw.get("outputs", {})
    outputs = OutputsConfig(results_root=_resolve_repo_path(str(out_raw.get("results_root", "results"))))

    return AppConfig(
        docker=docker,
        inputs=inputs,
        whitelist=whitelist,
        parallelism=parallelism,
        outputs=outputs,
    )


def _load_two_column_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        k, v = parts[0].strip(), parts[1].strip()
        if k:
            mapping[k] = v
    return mapping


def _seq_identity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if len(a) == len(b):
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / float(len(a)) if a else 0.0
    # Fallback: normalize by max length (no gaps modeled)
    n = min(len(a), len(b))
    matches = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    return matches / float(max(len(a), len(b)))


def _hungarian_max(weights: List[List[float]]) -> List[int]:
    """
    Solve max weight assignment for a square matrix.
    Returns assignment list: row i -> col assignment[i].
    """
    n = len(weights)
    if n == 0:
        return []
    for row in weights:
        if len(row) != n:
            raise ValueError("weights must be a square matrix")

    max_w = max(max(r) for r in weights)
    cost = [[max_w - w for w in row] for row in weights]  # minimization

    m = n
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    INF = 10**18
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float(INF)] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float(INF)
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _load_explicit_chain_mapping(
    target_dir: Path,
    cfg: InputsConfig,
    ref_path: Path,
    model_path: Path,
) -> Optional[Tuple[Path, Dict[str, str]]]:
    """
    Optional overrides:
      - <target>/chain_mappings/<ref_basename>__<model_basename>.json
      - <target>/chain_mappings/<ref_stem>__<model_stem>.json
      - <target>/chain_mapping.json
    File content: {"A": "B", "B": "A"} where keys are model chain IDs and values are reference chain IDs.
    """
    chain_dir = target_dir / cfg.chain_mapping_dirname
    candidates: List[Path] = []
    if chain_dir.is_dir():
        candidates.append(chain_dir / f"{ref_path.name}__{model_path.name}.json")
        candidates.append(chain_dir / f"{ref_path.stem}__{model_path.stem}.json")
    candidates.append(target_dir / cfg.chain_mapping_filename)

    for p in candidates:
        if not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Bad chain mapping JSON: {p} ({e})")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"Chain mapping must be a non-empty JSON object: {p}")
        mapping: Dict[str, str] = {}
        for k, v in raw.items():
            ks = str(k).strip()
            vs = str(v).strip()
            if len(ks) != 1 or len(vs) != 1:
                raise ValueError(f"Chain IDs must be 1-char strings: {p} ({ks!r}->{vs!r})")
            mapping[ks] = vs
        return (p, mapping)
    return None


def _auto_chain_mapping(
    ref: FilterResult,
    model: FilterResult,
) -> ChainMappingResult:
    errors: List[str] = []
    warnings: List[str] = []

    ref_chains = sorted(ref.chains.keys())
    model_chains = sorted(model.chains.keys())

    if len(ref_chains) != len(model_chains):
        return ChainMappingResult(
            ok=False,
            mapping={},
            source="auto",
            errors=(f"Chain count mismatch: ref={len(ref_chains)} model={len(model_chains)}",),
            warnings=(),
        )

    n = len(ref_chains)
    sim: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, mc in enumerate(model_chains):
        for j, rc in enumerate(ref_chains):
            sim[i][j] = _seq_identity(model.chains[mc].sequence, ref.chains[rc].sequence)

    assignment = _hungarian_max(sim)
    if any(x < 0 for x in assignment):
        errors.append("Failed to compute a complete chain assignment")
        return ChainMappingResult(
            ok=False,
            mapping={},
            source="auto",
            errors=tuple(errors),
            warnings=tuple(warnings),
            similarity_matrix=sim,
        )

    mapping: Dict[str, str] = {}
    for i, j in enumerate(assignment):
        mapping[model_chains[i]] = ref_chains[j]

    # Simple ambiguity warnings: low identity or near-ties
    for i, mc in enumerate(model_chains):
        row = sim[i]
        best = max(row) if row else 0.0
        best_j = assignment[i]
        second = max((v for jj, v in enumerate(row) if jj != best_j), default=0.0)
        if best < 0.95:
            warnings.append(f"Low chain identity for model chain '{mc}': best={best:.3f}")
        if (best - second) < 0.02 and n > 1:
            warnings.append(
                f"Ambiguous mapping for model chain '{mc}': best={best:.3f}, second={second:.3f}"
            )

    return ChainMappingResult(
        ok=True,
        mapping=mapping,
        source="auto",
        errors=tuple(errors),
        warnings=tuple(warnings),
        similarity_matrix=sim,
    )


def _validate_chain_mapping_strict(
    ref: FilterResult,
    model: FilterResult,
    mapping: Dict[str, str],
) -> Tuple[bool, List[str]]:
    """
    Strict contract:
      - chains are bijective
      - for each mapped chain, sequences must be identical
      - residue numbering (resSeq/iCode) must be identical
    """
    errors: List[str] = []

    ref_chains = set(ref.chains.keys())
    model_chains = set(model.chains.keys())

    if set(mapping.keys()) != model_chains:
        errors.append(f"Chain mapping keys must match model chains: {sorted(model_chains)}")
    if set(mapping.values()) != ref_chains:
        errors.append(f"Chain mapping values must match reference chains: {sorted(ref_chains)}")

    for mc, rc in mapping.items():
        if mc not in model.chains or rc not in ref.chains:
            continue
        mci = model.chains[mc]
        rci = ref.chains[rc]
        if mci.sequence != rci.sequence:
            errors.append(
                f"Sequence mismatch for chain {mc}->{rc}: "
                f"len(model)={len(mci.sequence)} len(ref)={len(rci.sequence)}"
            )
        if mci.residues != rci.residues:
            errors.append(
                f"Residue numbering mismatch for chain {mc}->{rc}: "
                f"len(model)={len(mci.residues)} len(ref)={len(rci.residues)}"
            )
    return (len(errors) == 0, errors)


def _rewrite_model_chains(
    model_filtered: Path,
    out_path: Path,
    mapping: Dict[str, str],
    ref_chain_order: Sequence[str],
) -> None:
    """
    Rewrite chain IDs in a filtered model PDB to match reference chain IDs, and reorder chains.
    """
    atoms: List[Tuple[int, str, int, str, str]] = []
    # (order_idx, line, resSeq, iCode, new_chain)
    for idx, raw in enumerate(model_filtered.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if not raw.startswith("ATOM"):
            continue
        if len(raw) < 27:
            continue
        old_chain = raw[21]
        new_chain = mapping.get(old_chain)
        if new_chain is None:
            # Skip atoms from unexpected chains; strict validation should catch this earlier.
            continue
        try:
            res_seq = int(raw[22:26])
        except Exception:
            res_seq = 0
        i_code = raw[26]
        line = raw[:21] + new_chain + raw[22:]
        order_idx = ref_chain_order.index(new_chain) if new_chain in ref_chain_order else 10**6
        atoms.append((order_idx, line, res_seq, i_code, new_chain))

    atoms.sort(key=lambda t: (t[0], t[2], t[3], t[1]))
    out_lines: List[str] = []
    last_chain: Optional[str] = None
    for _ord, line, _res_seq, _ic, ch in atoms:
        if last_chain is not None and ch != last_chain:
            out_lines.append("TER")
        out_lines.append(line)
        last_chain = ch
    if out_lines:
        out_lines.append("TER")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _docker_run(
    *,
    image: str,
    platform: str,
    mount_dir: Path,
    workdir: str,
    args: Sequence[str],
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    """
    Run a docker container mounting mount_dir -> /work.
    No auto-pull: images are checked upfront.
    """
    cmd: List[str] = ["docker", "run", "--rm", "--platform", platform]
    try:
        uid = os.getuid()
        gid = os.getgid()
        cmd += ["-u", f"{uid}:{gid}"]
    except Exception:
        pass

    cmd += ["-v", f"{str(mount_dir)}:/work:rw", "-w", workdir]
    if extra_env:
        for k, v in extra_env.items():
            cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    cmd.extend(list(args))

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", check=False)
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(p.stdout or "", encoding="utf-8")
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(p.stderr or "", encoding="utf-8")
    return int(p.returncode)


def _metric_error_result(metric: str, error: str, **extra: object) -> Dict[str, object]:
    data: Dict[str, object] = {"metric": metric, "ok": False, "error": error}
    data.update(extra)
    return data


def _compute_rmsd(
    *,
    ref_pdb: Path,
    model_pdb: Path,
    whitelist_atoms: Tuple[str, ...],
    pvalue_param: str = "-",
) -> Dict[str, object]:
    try:
        from Bio.PDB import PDBParser, Superimposer  # type: ignore
    except Exception as e:
        return _metric_error_result("rmsd", f"Missing Biopython dependency: {e}")

    import math

    def erf(z: float) -> float:
        # match legacy implementation behavior (avoid platform differences)
        t = 1.0 / (1.0 + 0.5 * abs(z))
        ans = 1 - t * math.exp(
            -z * z
            - 1.26551223
            + t
            * (
                1.00002368
                + t
                * (
                    0.37409196
                    + t
                    * (
                        0.09678418
                        + t
                        * (
                            -0.18628806
                            + t
                            * (
                                0.27886807
                                + t
                                * (
                                    -1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * (0.17087277)))
                                )
                            )
                        )
                    )
                )
            )
        )
        return ans if z >= 0.0 else -ans

    def pvalue(m: float, n_res: int, param: str) -> float:
        if param == "+":
            a, b = 5.1, 15.8
        elif param == "-":
            a, b = 6.4, 12.7
        else:
            raise ValueError("PVALUE must be '+' or '-'")
        rmsd0 = a * (n_res**0.41) - b
        z = (m - rmsd0) / 1.8
        return (1.0 + erf(z / (2**0.5))) / 2.0

    parser = PDBParser(QUIET=True)
    try:
        ref_struct = parser.get_structure("ref", str(ref_pdb))[0]
        mdl_struct = parser.get_structure("mdl", str(model_pdb))[0]
    except Exception as e:
        return _metric_error_result("rmsd", f"Failed to parse PDB: {e}")

    wl = set(whitelist_atoms)
    ref_atoms = []
    mdl_atoms = []
    missing = 0
    used = 0
    residues_count = 0

    for ref_chain in ref_struct:
        chain_id = ref_chain.id
        if chain_id not in mdl_struct:
            return _metric_error_result("rmsd", f"Missing chain '{chain_id}' in model")
        mdl_chain = mdl_struct[chain_id]
        for ref_res in ref_chain:
            if ref_res.id[0].strip():
                continue
            residues_count += 1
            if ref_res.id not in mdl_chain:
                return _metric_error_result(
                    "rmsd",
                    f"Missing residue {chain_id}:{ref_res.id[1]}{ref_res.id[2].strip()} in model",
                )
            mdl_res = mdl_chain[ref_res.id]
            for atom_name in whitelist_atoms:
                if atom_name not in wl:
                    continue
                if atom_name in ref_res and atom_name in mdl_res:
                    ref_atoms.append(ref_res[atom_name])
                    mdl_atoms.append(mdl_res[atom_name])
                    used += 1
                else:
                    missing += 1

    if used < 3:
        return _metric_error_result("rmsd", f"Too few atom pairs for superposition: {used}")

    sup = Superimposer()
    sup.set_atoms(ref_atoms, mdl_atoms)
    rmsd_val = float(sup.rms)

    try:
        pv = float(pvalue(rmsd_val, residues_count, pvalue_param))
    except Exception as e:
        pv = float("nan")

    return {
        "metric": "rmsd",
        "ok": True,
        "rmsd": rmsd_val,
        "pvalue": pv,
        "pvalue_param": pvalue_param,
        "n_residues": residues_count,
        "atoms_used": used,
        "atoms_missing": missing,
    }


def _parse_mc_annotate_mcout(mcout_path: Path) -> List[Tuple[str, str, int, str]]:
    """
    Parse MC-Annotate .mcout file and return canonical interactions:
      - (type, end1, end2, extra) where end is "chain:pos" and end1 <= end2
      - type in {"PAIR_2D","PAIR_3D","STACK"}
    """
    import re

    interactions: List[Tuple[str, str, int, str, str, int, str, str, str, str]] = []

    STATE_OUT = 0
    STATE_PAIR = 2
    STATE_STACK = 3

    pattern_pair = r"^([A-Z]|\'[0-9]\'|)(\d+)-([A-Z]|\'[0-9]\'|)(\d+) : (\w+)-(\w+) ([\w\']+)/([\w\']+)(?:.*)pairing( (parallel|antiparallel) (cis|trans))"
    pattern_stack = r"^([A-Z]|\'[0-9]\'|)(\d+)-([A-Z]|\'[0-9]\'|)(\d+) :.*(inward|upward|downward|outward).*"

    def convert_pair(match) -> Optional[Tuple[str, str, int, str, str, int, str, str]]:
        int_a = match[6][0].upper()
        int_b = match[7][0].upper()
        if (int_a not in ["W", "H", "S"]) or (int_b not in ["W", "H", "S"]):
            return None
        chain_a = match[0].replace("'", "")
        pos_a = int(match[1])
        nt_a = match[4]
        chain_b = match[2].replace("'", "")
        pos_b = int(match[3])
        nt_b = match[5]
        int_type = f"{int_a}{int_b}"
        int_orientation = match[10].lower()
        pair_name = "PAIR_2D" if (f"{int_orientation}{int_a}{int_b}" == "cisWW") else "PAIR_3D"

        a = (chain_a, pos_a, nt_a)
        b = (chain_b, pos_b, nt_b)
        if ((chain_a == chain_b) and (pos_a < pos_b)) or (chain_a < chain_b):
            c1, p1, n1 = a
            c2, p2, n2 = b
        else:
            c1, p1, n1 = b
            c2, p2, n2 = a
        extra = f"{int_type}{int_orientation}"
        return (pair_name, c1, p1, c2, p2, extra, n1, n2)

    def convert_stack(match) -> Tuple[str, str, int, str, int, str]:
        chain_a = match[0].replace("'", "")
        pos_a = int(match[1])
        chain_b = match[2].replace("'", "")
        pos_b = int(match[3])
        int_type = match[4]
        # canonical endpoint order
        if ((chain_a == chain_b) and (pos_a < pos_b)) or (chain_a < chain_b):
            c1, p1, c2, p2 = chain_a, pos_a, chain_b, pos_b
        else:
            c1, p1, c2, p2 = chain_b, pos_b, chain_a, pos_a
        return ("STACK", c1, p1, c2, p2, int_type)

    state = STATE_OUT
    model_count = 0
    for line in mcout_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        if "one_hbond" in s:
            continue
        if s.startswith("Residue conformations"):
            if model_count == 0:
                model_count += 1
                continue
            break
        if s.startswith("Base-pairs"):
            state = STATE_PAIR
            continue
        if s.startswith("Adjacent stackings") or s.startswith("Non-Adjacent stackings"):
            state = STATE_STACK
            continue
        if s.endswith("----------"):
            state = STATE_OUT
            continue

        if state == STATE_PAIR:
            m = re.match(pattern_pair, s)
            if m:
                g = m.groups()
                conv = convert_pair(g)
                if conv is None:
                    continue
                pair_name, c1, p1, c2, p2, extra, _n1, _n2 = conv
                interactions.append((pair_name, c1, p1, "", c2, p2, "", extra, "", ""))
        elif state == STATE_STACK:
            m = re.match(pattern_stack, s)
            if m:
                g = m.groups()
                t, c1, p1, c2, p2, extra = convert_stack(g)
                interactions.append((t, c1, p1, "", c2, p2, "", extra, "", ""))

    canon: List[Tuple[str, str, int, str]] = []
    for (t, c1, p1, _nt1, c2, p2, _nt2, extra, _e2, _e3) in interactions:
        end1 = f"{c1}:{p1}"
        end2 = f"{c2}:{p2}"
        if end2 < end1:
            end1, end2 = end2, end1
        canon.append((t, end1, end2, extra))
    return canon


def _inf_score(src: List[Tuple[str, str, str, str]], trg: List[Tuple[str, str, str, str]]) -> Tuple[float, int, int, int]:
    """
    Returns (INF, TP, FP, FN) following puzzlesBenchmark_after/pdb_utils.py semantics.
    """
    src_pairs = set(src)
    trg_pairs = set(trg)
    if not src_pairs and not trg_pairs:
        return (-1.0, 0, 0, 0)
    common = src_pairs & trg_pairs
    tp = len(common)
    fn = len(src_pairs - trg_pairs)
    fp = len(trg_pairs - src_pairs)
    if tp == 0 and (fp == 0 or fn == 0):
        return (-1.0, tp, fp, fn)
    ppv = tp / float(tp + fp)
    sty = tp / float(tp + fn)
    return (((ppv * sty) ** 0.5), tp, fp, fn)


def _compute_inf(
    *,
    pair_dir: Path,
    ref_pdb: Path,
    model_pdb: Path,
    docker: DockerConfig,
) -> Dict[str, object]:
    inf_dir = pair_dir / "inf"
    inf_dir.mkdir(parents=True, exist_ok=True)
    stdout_ref = inf_dir / "mcannotate.ref.stdout.txt"
    stderr_ref = inf_dir / "mcannotate.ref.stderr.txt"
    stdout_m = inf_dir / "mcannotate.model.stdout.txt"
    stderr_m = inf_dir / "mcannotate.model.stderr.txt"
    mcout_ref = inf_dir / f"{ref_pdb.name}.mcout"
    mcout_m = inf_dir / f"{model_pdb.name}.mcout"

    rc1 = _docker_run(
        image=docker.tools_image,
        platform=docker.platform,
        mount_dir=pair_dir,
        workdir="/work",
        args=["/opt/tools/MC-Annotate", str(ref_pdb.relative_to(pair_dir))],
        stdout_path=stdout_ref,
        stderr_path=stderr_ref,
    )
    if rc1 != 0:
        return _metric_error_result("inf", f"MC-Annotate failed for reference (rc={rc1})")
    mcout_ref.write_text(stdout_ref.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    rc2 = _docker_run(
        image=docker.tools_image,
        platform=docker.platform,
        mount_dir=pair_dir,
        workdir="/work",
        args=["/opt/tools/MC-Annotate", str(model_pdb.relative_to(pair_dir))],
        stdout_path=stdout_m,
        stderr_path=stderr_m,
    )
    if rc2 != 0:
        return _metric_error_result("inf", f"MC-Annotate failed for model (rc={rc2})")
    mcout_m.write_text(stdout_m.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    try:
        ref_int = _parse_mc_annotate_mcout(mcout_ref)
        mdl_int = _parse_mc_annotate_mcout(mcout_m)
    except Exception as e:
        return _metric_error_result("inf", f"Failed to parse mcout: {e}")

    def filt(items, types: Tuple[str, ...]) -> List[Tuple[str, str, str, str]]:
        return [(t, a, b, ex) for (t, a, b, ex) in items if t in types]

    inf_all, tp_all, fp_all, fn_all = _inf_score(ref_int, mdl_int)
    inf_wc, tp_wc, fp_wc, fn_wc = _inf_score(filt(ref_int, ("PAIR_2D",)), filt(mdl_int, ("PAIR_2D",)))
    inf_nwc, tp_nwc, fp_nwc, fn_nwc = _inf_score(filt(ref_int, ("PAIR_3D",)), filt(mdl_int, ("PAIR_3D",)))
    inf_stack, tp_st, fp_st, fn_st = _inf_score(filt(ref_int, ("STACK",)), filt(mdl_int, ("STACK",)))

    return {
        "metric": "inf",
        "ok": True,
        "INF_ALL": inf_all,
        "INF_WC": inf_wc,
        "INF_NWC": inf_nwc,
        "INF_STACK": inf_stack,
        "counts": {
            "ALL": {"TP": tp_all, "FP": fp_all, "FN": fn_all},
            "WC": {"TP": tp_wc, "FP": fp_wc, "FN": fn_wc},
            "NWC": {"TP": tp_nwc, "FP": fp_nwc, "FN": fn_nwc},
            "STACK": {"TP": tp_st, "FP": fp_st, "FN": fn_st},
        },
        "mcout": {"ref": str(mcout_ref.relative_to(pair_dir)), "model": str(mcout_m.relative_to(pair_dir))},
    }


def _compute_tm(*, pair_dir: Path, ref_pdb: Path, model_pdb: Path, docker: DockerConfig) -> Dict[str, object]:
    tm_dir = pair_dir / "tm"
    tm_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = tm_dir / "rnaalign.stdout.txt"
    stderr_path = tm_dir / "rnaalign.stderr.txt"

    rc = _docker_run(
        image=docker.tools_image,
        platform=docker.platform,
        mount_dir=pair_dir,
        workdir="/work",
        args=[
            "/opt/tools/RNAalign/RNAalign",
            str(model_pdb.relative_to(pair_dir)),
            str(ref_pdb.relative_to(pair_dir)),
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if rc != 0:
        return _metric_error_result("tm", f"RNAalign failed (rc={rc})", returncode=rc)

    import re

    txt = stdout_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"TM-score\s*=\s*([0-1]\.\d+)", txt)
    if not m:
        # fallback: any 0.x float
        m = re.search(r"\b([0-1]\.\d{2,})\b", txt)
    if not m:
        return _metric_error_result("tm", "TM-score not found in RNAalign output", returncode=rc)
    try:
        val = float(m.group(1))
    except Exception:
        return _metric_error_result("tm", f"TM-score not parseable: {m.group(1)!r}", returncode=rc)

    return {"metric": "tm", "ok": True, "tm": val, "returncode": rc}


def _compute_mcq(*, pair_dir: Path, ref_pdb: Path, model_pdb: Path, docker: DockerConfig) -> Dict[str, object]:
    mcq_dir = pair_dir / "mcq"
    tmp_dir = mcq_dir / "tmp"
    mcq_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = mcq_dir / "mcq.stdout.txt"
    stderr_path = mcq_dir / "mcq.stderr.txt"

    jar = "/opt/tools/mcq-cli/mcq-cli/target/mcq-cli-1.6-SNAPSHOT-jar-with-dependencies.jar"
    cls = "pl.poznan.put.mcq.cli.Global"

    rc = _docker_run(
        image=docker.tools_image,
        platform=docker.platform,
        mount_dir=pair_dir,
        workdir="/work",
        args=[
            "java",
            f"-Djava.io.tmpdir=/work/{mcq_dir.relative_to(pair_dir) / 'tmp'}",
            "-cp",
            jar,
            cls,
            str(model_pdb.relative_to(pair_dir)),
            str(ref_pdb.relative_to(pair_dir)),
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if rc != 0:
        return _metric_error_result("mcq", f"MCQ failed (rc={rc})", returncode=rc)

    # Find matrix.csv under mcq/tmp
    matrices = sorted(tmp_dir.rglob("matrix.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matrices:
        return _metric_error_result("mcq", "matrix.csv not found under mcq/tmp", returncode=rc)
    matrix = matrices[0]

    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(matrix, index_col=0, sep=",")
        val = float(df.iloc[0, 1])
    except Exception as e:
        return _metric_error_result("mcq", f"Failed to parse matrix.csv: {e}", returncode=rc)

    return {
        "metric": "mcq",
        "ok": True,
        "mcq": val,
        "returncode": rc,
        "matrix_csv": str(matrix.relative_to(pair_dir)),
    }


def _compute_gdt(*, pair_dir: Path, ref_pdb: Path, model_pdb: Path, docker: DockerConfig) -> Dict[str, object]:
    import shutil
    import re

    gdt_dir = pair_dir / "gdt"
    gdt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = gdt_dir / "lga.stdout.txt"
    stderr_path = gdt_dir / "lga.stderr.txt"

    ref_bn = "ref.pdb"
    model_bn = "model.pdb"
    shutil.copy2(ref_pdb, gdt_dir / ref_bn)
    shutil.copy2(model_pdb, gdt_dir / model_bn)

    mol2process = f"GDT.{model_bn}.{ref_bn}"
    inner = f"""set -e
mkdir -p bin MOL2 RESULTS TMP
[ -e bin/collect_PDB.pl ] || ln -sf /opt/lga/collect_PDB.pl bin/collect_PDB.pl
[ -e bin/lga ] || ln -sf /opt/lga/lga bin/lga
export PATH=/work/bin:/opt/lga:$PATH
/opt/lga/runlga.mol_mol.pl {model_bn} {ref_bn} -4 -d:4 -atom:C4, -stral -o2
/opt/lga/collect_PDB.pl {model_bn} > MOL2/{mol2process}
/opt/lga/collect_PDB.pl {ref_bn} >> MOL2/{mol2process}
/opt/lga/lga -4 -d:4.0 -o0 -atom:C4 -lga_m -stral -ie {mol2process} > RESULTS/{mol2process}.res
grep '^LGA ' RESULTS/{mol2process}.res >> MOL2/{mol2process} || true
/opt/lga/lga -3 -sia -d:5.0 -atom:C4 -o2 -al -ie {mol2process} > RESULTS/{mol2process}.gdt_res
"""

    rc = _docker_run(
        image=docker.gdt_image,
        platform=docker.platform,
        mount_dir=gdt_dir,
        workdir="/work",
        args=["bash", "-lc", inner],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if rc != 0:
        return _metric_error_result("gdt", f"LGA/GDT failed (rc={rc})", returncode=rc)

    out_txt = gdt_dir / "gdt.txt"
    gdt_res = gdt_dir / "RESULTS" / f"{mol2process}.gdt_res"
    res = gdt_dir / "RESULTS" / f"{mol2process}.res"

    # Reuse the post-processing logic from gdt_docker.py to compute GDT_TS/HA
    N1 = N2 = NB = None
    v1 = v2 = None
    if res.exists():
        for line in res.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "SUMMARY(LGA)" in line:
                toks = line.replace("#", " ").split()
                try:
                    i = toks.index("SUMMARY(LGA)")
                    N1 = float(toks[i + 1])
                    N2 = float(toks[i + 2])
                except Exception:
                    pass
                break

    if gdt_res.exists():
        for line in gdt_res.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "SUMMARY(GDT)" in line:
                toks = line.replace("#", " ").split()
                try:
                    j = toks.index("SUMMARY(GDT)")
                    NB = float(toks[j + 2])
                except Exception:
                    pass
            if "GDT PERCENT_AT" in line:
                t = line.split()
                try:
                    v1 = (float(t[2]) + float(t[3]) + float(t[5]) + float(t[9])) / 4.0
                    v2 = (float(t[3]) + float(t[5]) + float(t[9]) + float(t[17])) / 4.0
                except Exception:
                    pass

    gdt_ha = gdt_ts = None
    if N1 and NB and v1 is not None and v2 is not None:
        try:
            gdt_ha = v1 * NB / N1
            gdt_ts = v2 * NB / N1
        except Exception:
            pass

    # write a small human-readable file (optional)
    txt = f"GDT_HA = {gdt_ha if gdt_ha is not None else 0.0:6.2f}  GDT_TS = {gdt_ts if gdt_ts is not None else 0.0:6.2f}\n"
    out_txt.write_text(txt, encoding="utf-8")

    return {
        "metric": "gdt",
        "ok": True,
        "GDT_HA": float(gdt_ha) if gdt_ha is not None else None,
        "GDT_TS": float(gdt_ts) if gdt_ts is not None else None,
        "returncode": rc,
        "params": {"atom": "C4", "d_lga": 4.0, "d_gdt": 5.0},
    }


def _ensure_ost_script(dst_dir: Path) -> Path:
    script_src = REPO_ROOT / "puzzlesBatchEval" / "tools" / "ost_lddt.py"
    if not script_src.exists():
        raise FileNotFoundError(f"Missing OST lDDT script: {script_src}")
    dst = dst_dir / "ost_lddt.py"
    if not dst.exists():
        import shutil

        shutil.copy2(script_src, dst)
    return dst


def _compute_lddt(*, pair_dir: Path, ref_pdb: Path, model_pdb: Path, docker: DockerConfig) -> Dict[str, object]:
    import shutil

    lddt_dir = pair_dir / "lddt"
    lddt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = lddt_dir / "ost.stdout.txt"
    stderr_path = lddt_dir / "ost.stderr.txt"

    # Copy inputs locally (keep the lDDT run folder self-contained)
    ref_bn = "ref.pdb"
    model_bn = "model.pdb"
    shutil.copy2(ref_pdb, lddt_dir / ref_bn)
    shutil.copy2(model_pdb, lddt_dir / model_bn)

    # Identity chain mapping after chain renaming
    # Derive chain IDs from filtered reference PDB by scanning column 22
    chains = sorted({ln[21] for ln in (lddt_dir / ref_bn).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.startswith("ATOM") and len(ln) > 21})
    chain_map = {c: c for c in chains if c.strip()}
    mapping_path = lddt_dir / "chain_mapping.json"
    _write_json(mapping_path, chain_map)

    out_json = lddt_dir / "lddt.json"
    script = _ensure_ost_script(lddt_dir)

    rc = _docker_run(
        image=docker.ost_image,
        platform=docker.platform,
        mount_dir=lddt_dir,
        workdir="/work",
        args=[script.name, model_bn, ref_bn, mapping_path.name, out_json.name],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if rc != 0:
        return _metric_error_result("lddt", f"OpenStructure lDDT failed (rc={rc})", returncode=rc)

    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        val = float(data.get("lDDT"))
    except Exception as e:
        return _metric_error_result("lddt", f"Failed to parse lddt.json: {e}", returncode=rc)

    return {
        "metric": "lddt",
        "ok": True,
        "lddt": val,
        "returncode": rc,
        "out_json": str(out_json.relative_to(pair_dir)),
    }


def _parse_clashscore(molprobity_txt: Path, model_basename: str) -> Optional[float]:
    # Match line by base filename (strip optional FH suffix)
    for raw in molprobity_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#"):
            continue
        parts = line.split(":")
        name = parts[0].strip()
        base = os.path.basename(name)
        if base.endswith("FH.pdb"):
            base_core = base[:-6]
        elif base.endswith(".pdb"):
            base_core = base[:-4]
        else:
            base_core = base
        if base_core != model_basename.replace(".pdb", ""):
            continue
        # clashscore: prefer canonical position, fallback scan
        if len(parts) > 8:
            try:
                return float(parts[8].strip().split()[0])
            except Exception:
                pass
        for p in reversed(parts):
            token = p.strip().split()[0] if p.strip() else ""
            try:
                return float(token)
            except Exception:
                continue
    return None


def _compute_clash(*, pair_dir: Path, ref_pdb: Path, model_pdb: Path, docker: DockerConfig) -> Dict[str, object]:
    import shutil

    clash_dir = pair_dir / "clash"
    inp_dir = clash_dir / "input"
    out_dir = clash_dir / "reduced"
    tmp_dir = clash_dir / "tmp"
    clash_dir.mkdir(parents=True, exist_ok=True)
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ref_bn = "ref.pdb"
    model_bn = "model.pdb"
    shutil.copy2(ref_pdb, inp_dir / ref_bn)
    shutil.copy2(model_pdb, inp_dir / model_bn)

    molprobity_txt = clash_dir / "molprobity.txt"
    stdout_path = clash_dir / "molprobity.stdout.txt"
    stderr_path = clash_dir / "molprobity.stderr.txt"

    php_prefix = (
        "env SERVER_NAME=localhost HTTP_HOST=localhost php "
        "-d session.save_path=/work/tmp "
        "-d session.auto_start=1 "
        "-d session.use_cookies=0 "
        "-d session.use_only_cookies=0 "
        "-d session.cache_limiter= "
        "-d variables_order=EGPCS "
        "-d register_argc_argv=On"
    )
    mp_dir = "/opt/tools/MolProbity/cmdline"
    inner = f"""set -e
{php_prefix} {mp_dir}/reduce-build input reduced
{php_prefix} {mp_dir}/oneline-analysis -nocbeta -norota -norama reduced > molprobity.txt
"""

    rc = _docker_run(
        image=docker.tools_image,
        platform=docker.platform,
        mount_dir=clash_dir,
        workdir="/work",
        args=["bash", "-lc", inner],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if rc != 0:
        return _metric_error_result("clash", f"MolProbity failed (rc={rc})", returncode=rc)
    if not molprobity_txt.exists() or molprobity_txt.stat().st_size == 0:
        return _metric_error_result("clash", "MolProbity output missing/empty", returncode=rc)

    val = _parse_clashscore(molprobity_txt, model_bn)
    if val is None:
        return _metric_error_result("clash", "Failed to extract clashscore from molprobity.txt", returncode=rc)

    return {
        "metric": "clash",
        "ok": True,
        "clashscore": float(val),
        "returncode": rc,
        "molprobity_txt": str(molprobity_txt.relative_to(pair_dir)),
    }


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rank_series(values_by_key: Dict[str, float], *, ascending: bool) -> Dict[str, int]:
    """
    Rank values (1..N) with 'min' tie strategy. Missing keys are not included.
    """
    items = sorted(values_by_key.items(), key=lambda kv: kv[1], reverse=not ascending)
    ranks: Dict[str, int] = {}
    last_val: Optional[float] = None
    last_rank = 0
    for idx, (k, v) in enumerate(items, start=1):
        if last_val is None or v != last_val:
            last_rank = idx
            last_val = v
        ranks[k] = last_rank
    return ranks


def _summarize_target(out_target: Path) -> None:
    """
    Build per-target tables:
      - tables/summary_all_pairs.csv
      - tables/errors.csv
    """
    import csv
    import math

    target_manifest = _read_json(out_target / "manifest.json") or {}
    pairs = target_manifest.get("pairs") or []

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    pair_dir_by_id: Dict[str, Path] = {}

    for p in pairs:
        pair_dir = Path(str(p.get("pair_dir", "")))
        if not pair_dir.is_absolute():
            pair_dir = (out_target / pair_dir).resolve()
        pm = _read_json(pair_dir / "manifest.json") or {}
        ref_name = Path(str(pm.get("reference_input", ""))).name
        model_name = Path(str(pm.get("model_input", ""))).name
        pair_id = f"{ref_name}__{model_name}"
        pair_dir_by_id[pair_id] = pair_dir

        row: Dict[str, object] = {
            "pair": pair_id,
            "reference": ref_name,
            "model": model_name,
        }
        pair_errors: List[Dict[str, str]] = []
        metrics_blob: Dict[str, object] = {}

        # ---------- RMSD (+ p-value) ----------
        rmsd_js = _read_json(pair_dir / "rmsd" / "result.json")
        if not rmsd_js or not bool(rmsd_js.get("ok")):
            pair_errors.append(
                {"pair": pair_id, "metric": "rmsd", "error": str((rmsd_js or {}).get("error", "missing"))}
            )
        else:
            row["RMSD"] = rmsd_js.get("rmsd")
            row["p-value"] = rmsd_js.get("pvalue")
        if rmsd_js:
            metrics_blob["rmsd"] = rmsd_js

        # ---------- INF ----------
        inf_js = _read_json(pair_dir / "inf" / "result.json")
        if not inf_js or not bool(inf_js.get("ok")):
            pair_errors.append(
                {"pair": pair_id, "metric": "inf", "error": str((inf_js or {}).get("error", "missing"))}
            )
        else:
            row["INF_all"] = inf_js.get("INF_ALL")
            row["INF_wc"] = inf_js.get("INF_WC")
            row["INF_nwc"] = inf_js.get("INF_NWC")
            row["INF_stack"] = inf_js.get("INF_STACK")
        if inf_js:
            metrics_blob["inf"] = inf_js

        # ---------- DI (derived) ----------
        try:
            rmsd_val = float(row["RMSD"]) if row.get("RMSD") is not None else None
            inf_all = float(row["INF_all"]) if row.get("INF_all") is not None else None
            if rmsd_val is None or inf_all is None:
                raise ValueError("missing RMSD or INF_all")
            row["DI"] = abs(rmsd_val / inf_all) if inf_all != 0.0 else float("inf")
        except Exception as e:
            pair_errors.append({"pair": pair_id, "metric": "DI", "error": f"derive failed: {e}"})
        metrics_blob["DI"] = row.get("DI")

        # ---------- TM ----------
        tm_js = _read_json(pair_dir / "tm" / "result.json")
        if not tm_js or not bool(tm_js.get("ok")):
            pair_errors.append(
                {"pair": pair_id, "metric": "TM", "error": str((tm_js or {}).get("error", "missing"))}
            )
        else:
            row["TM"] = tm_js.get("tm")
        if tm_js:
            metrics_blob["tm"] = tm_js

        # ---------- MCQ ----------
        mcq_js = _read_json(pair_dir / "mcq" / "result.json")
        if not mcq_js or not bool(mcq_js.get("ok")):
            pair_errors.append(
                {"pair": pair_id, "metric": "MCQ", "error": str((mcq_js or {}).get("error", "missing"))}
            )
        else:
            row["MCQ"] = mcq_js.get("mcq")
        if mcq_js:
            metrics_blob["mcq"] = mcq_js

        # ---------- GDT ----------
        gdt_js = _read_json(pair_dir / "gdt" / "result.json")
        if not gdt_js or not bool(gdt_js.get("ok")) or gdt_js.get("GDT_TS") is None:
            pair_errors.append(
                {"pair": pair_id, "metric": "GDT", "error": str((gdt_js or {}).get("error", "missing"))}
            )
        else:
            row["GDT"] = gdt_js.get("GDT_TS")
        if gdt_js:
            metrics_blob["gdt"] = gdt_js

        # ---------- lDDT ----------
        lddt_js = _read_json(pair_dir / "lddt" / "result.json")
        if not lddt_js or not bool(lddt_js.get("ok")):
            pair_errors.append(
                {"pair": pair_id, "metric": "lDDT", "error": str((lddt_js or {}).get("error", "missing"))}
            )
        else:
            row["lDDT"] = lddt_js.get("lddt")
        if lddt_js:
            metrics_blob["lddt"] = lddt_js

        # ---------- clash ----------
        clash_js = _read_json(pair_dir / "clash" / "result.json")
        if not clash_js or not bool(clash_js.get("ok")):
            pair_errors.append(
                {"pair": pair_id, "metric": "clash", "error": str((clash_js or {}).get("error", "missing"))}
            )
        else:
            row["clash"] = clash_js.get("clashscore")
        if clash_js:
            metrics_blob["clash"] = clash_js

        rows.append(row)
        errors.extend(pair_errors)

        # Update pair manifest with a compact summary (one-stop view)
        try:
            pm["summary"] = {"row": row, "errors": pair_errors}
            pm["metrics"] = metrics_blob
            _write_json(pair_dir / "manifest.json", pm)
        except Exception:
            pass

    # Compute ranks (skip missing values)
    # Smaller is better: RMSD, DI, p-value, clash, MCQ
    # Larger is better: INF_*, TM, GDT, lDDT
    metrics_rank_spec = [
        ("RMSD", True),
        ("DI", True),
        ("p-value", True),
        ("clash", True),
        ("MCQ", True),
        ("INF_all", False),
        ("INF_wc", False),
        ("INF_nwc", False),
        ("INF_stack", False),
        ("TM", False),
        ("GDT", False),
        ("lDDT", False),
    ]

    # key: pair_id
    rows_by_pair = {str(r["pair"]): r for r in rows}
    for col, smaller_better in metrics_rank_spec:
        vals: Dict[str, float] = {}
        for pair_id, r in rows_by_pair.items():
            v = r.get(col)
            try:
                if v is None:
                    continue
                fv = float(v)
                if math.isnan(fv):
                    continue
                vals[pair_id] = fv
            except Exception:
                continue
        ranks = _rank_series(vals, ascending=smaller_better)
        rank_col = f"{col}_rank"
        for pair_id, r in rows_by_pair.items():
            if pair_id in ranks:
                r[rank_col] = ranks[pair_id]

    # Push rank-enriched rows back into per-pair manifests
    for r in rows:
        pid = str(r.get("pair", ""))
        pdir = pair_dir_by_id.get(pid)
        if not pdir:
            continue
        pm = _read_json(pdir / "manifest.json")
        if not pm:
            continue
        try:
            summary = pm.get("summary") if isinstance(pm.get("summary"), dict) else {}
            if isinstance(summary, dict):
                summary["row"] = r
                pm["summary"] = summary
            _write_json(pdir / "manifest.json", pm)
        except Exception:
            continue

    # Write CSVs
    tables_dir = out_target / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Keep column order stable
    cols = [
        "pair",
        "reference",
        "model",
        "RMSD",
        "RMSD_rank",
        "DI",
        "DI_rank",
        "INF_all",
        "INF_all_rank",
        "INF_wc",
        "INF_wc_rank",
        "INF_nwc",
        "INF_nwc_rank",
        "INF_stack",
        "INF_stack_rank",
        "clash",
        "clash_rank",
        "MCQ",
        "MCQ_rank",
        "TM",
        "TM_rank",
        "GDT",
        "GDT_rank",
        "lDDT",
        "lDDT_rank",
        "p-value",
        "p-value_rank",
    ]

    summary_csv = tables_dir / "summary_all_pairs.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    errors_csv = tables_dir / "errors.csv"
    with errors_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "metric", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)



def _filter_pdb(
    input_path: Path,
    output_path: Path,
    residue_map: Dict[str, str],
    atom_map: Dict[str, str],
    whitelist_atoms: Tuple[str, ...],
) -> FilterResult:
    """
    Apply rnapuzzles-style residue/atom name mapping + atom whitelist filtering.

    - Residue mapping uses residues.list (unknown => error)
    - Atom mapping uses atoms.list (unknown => error)
    - Atom synonyms are unified by atoms.list mapping before whitelist filtering
    - Only the first MODEL is processed (if MODEL/ENDMDL blocks exist)
    """
    whitelist = set(whitelist_atoms)
    errors: List[str] = []
    warnings: List[str] = []

    atoms_in = 0
    atoms_out = 0

    # residue_key -> base (A/C/G/U) and whether any atom survived whitelist
    residue_seen: Dict[Tuple[str, int, str], str] = {}
    residue_has_atom: Dict[Tuple[str, int, str], bool] = {}

    chains_order: List[str] = []

    # MODEL handling: if any MODEL record exists, only process the first model block
    model_count = 0
    in_first_model = False
    saw_model_records = False

    in_atom_block = False
    lines_out: List[str] = []

    try:
        text = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        return FilterResult(
            input_path=input_path,
            output_path=output_path,
            ok=False,
            errors=(f"Failed to read PDB: {e}",),
            warnings=(),
            chains={},
            atoms_in=0,
            atoms_out=0,
            residues_out=0,
        )

    for raw in text:
        if not raw:
            continue
        rec = raw[:6]
        if rec == "MODEL ":
            saw_model_records = True
            model_count += 1
            if model_count == 1:
                in_first_model = True
            else:
                # stop at 2nd model
                break
            continue
        if rec == "ENDMDL":
            if saw_model_records and in_first_model:
                break
            continue

        if saw_model_records and not in_first_model:
            continue

        if rec[:3] == "TER":
            if in_atom_block:
                lines_out.append("TER")
            in_atom_block = False
            continue

        if rec not in ("ATOM  ", "HETATM"):
            continue

        atoms_in += 1

        if len(raw) < 54:
            errors.append(f"Short ATOM/HETATM line (len<{54}): {raw}")
            continue

        try:
            serial = raw[6:11]
            name_in = raw[12:16].strip()
            alt_loc = raw[16]
            res_name_in = raw[17:20].strip()
            chain_id = raw[21]
            res_seq = int(raw[22:26])
            i_code = raw[26]
            x = raw[30:38]
            y = raw[38:46]
            z = raw[46:54]
            element = raw[76:78] if len(raw) >= 78 else ""
            charge = raw[78:80] if len(raw) >= 80 else ""
        except Exception as e:
            errors.append(f"Failed to parse ATOM/HETATM fields: {e} | {raw}")
            continue

        if element.strip() == "H":
            continue

        # Normalize chain id
        if chain_id == " ":
            chain_id = "A"

        # Residue mapping
        res_name_norm = residue_map.get(res_name_in)
        if res_name_norm is None:
            errors.append(f"Unknown residue name '{res_name_in}' in {input_path}")
            continue
        if res_name_norm == "-":
            continue

        # Track residue presence even if no whitelisted atoms survive,
        # so we can detect residues that would disappear after filtering.
        rkey = (chain_id, res_seq, i_code)
        if rkey not in residue_seen:
            residue_seen[rkey] = res_name_norm
            residue_has_atom[rkey] = False

        # Atom mapping (synonym normalization)
        name_norm = atom_map.get(name_in)
        if name_norm is None:
            errors.append(f"Unknown atom name '{name_in}' (res '{res_name_norm}') in {input_path}")
            continue
        if name_norm == "-":
            continue

        # Whitelist filtering
        if name_norm not in whitelist:
            continue

        residue_has_atom[rkey] = True

        if chain_id not in chains_order:
            chains_order.append(chain_id)

        # Normalize remaining PDB fields
        occupancy = "  1.00"
        temp_factor = "  0.00"
        element_out = element.strip() or name_norm[0]
        # PDB atom name field: keep to <=3 chars in this pipeline
        name_out = name_norm.ljust(3)

        line_out = (
            "ATOM  %5s  %3s%s%3s %s%4d%s   %8s%8s%8s%6s%6s		  %2s%2s"
            % (
                serial,
                name_out,
                alt_loc,
                res_name_norm,
                chain_id,
                res_seq,
                i_code,
                x,
                y,
                z,
                occupancy,
                temp_factor,
                element_out.rjust(2),
                charge.rjust(2),
            )
        )
        lines_out.append(line_out)
        atoms_out += 1
        in_atom_block = True

    if in_atom_block:
        lines_out.append("TER")

    # Any residue that survived residue mapping but has no whitelisted atoms would have been dropped.
    # Treat as an error because it changes residue count/numbering.
    dropped_residues = [k for k, has_atom in residue_has_atom.items() if not has_atom]
    if dropped_residues:
        errors.append(
            f"{len(dropped_residues)} residue(s) had no whitelisted atoms and were dropped: "
            + ", ".join([f"{c}:{n}{ic or ''}" for (c, n, ic) in dropped_residues[:10]])
            + (" ..." if len(dropped_residues) > 10 else "")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    except Exception as e:
        errors.append(f"Failed to write filtered PDB: {e}")

    # Build chain sequences from residues present in output (unique residue keys with has_atom True)
    residues_by_chain: Dict[str, List[Tuple[int, str, str]]] = {}
    for (c, n, ic), base in residue_seen.items():
        if not residue_has_atom.get((c, n, ic), False):
            continue
        residues_by_chain.setdefault(c, []).append((n, ic, base))
    chains: Dict[str, ChainInfo] = {}
    for c, items in residues_by_chain.items():
        items_sorted = sorted(items, key=lambda t: (t[0], t[1]))
        residues = tuple((n, ic) for (n, ic, _b) in items_sorted)
        seq = "".join(_b for (_n, _ic, _b) in items_sorted)
        chains[c] = ChainInfo(chain_id=c, sequence=seq, residues=residues)

    ok = len(errors) == 0
    return FilterResult(
        input_path=input_path,
        output_path=output_path,
        ok=ok,
        errors=tuple(errors),
        warnings=tuple(warnings),
        chains=chains,
        atoms_in=atoms_in,
        atoms_out=atoms_out,
        residues_out=sum(len(ci.residues) for ci in chains.values()),
    )


def _docker_image_exists(image: str) -> bool:
    try:
        p = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return p.returncode == 0
    except FileNotFoundError:
        raise RuntimeError("docker not found in PATH")


def _require_images(cfg: DockerConfig) -> None:
    missing = []
    for img in [cfg.tools_image, cfg.gdt_image, cfg.ost_image]:
        if not _docker_image_exists(img):
            missing.append(img)
    if missing:
        msg = "\n".join([f"- {x}" for x in missing])
        raise RuntimeError(
            "Required docker images are missing (will NOT auto-pull):\n" + msg
        )


def _iter_target_dirs(targets_root: Path) -> Iterable[Path]:
    for p in sorted(targets_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        yield p


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluator for targets/<target>/{references,models}/*.pdb (optionally processed/).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=(Path(__file__).resolve().parent / "config.json"))
    p.add_argument("--targets-root", type=Path, required=True)
    p.add_argument("--target", type=Path, action="append", default=[])
    p.add_argument("--run-id", default=None, help="If omitted, uses YYYYMMDD-HHMMSS.")
    p.add_argument("--results-root", type=Path, default=None, help="Override config.outputs.results_root")
    p.add_argument("--workers", type=int, default=None, help="Override config.parallelism.max_workers")
    p.add_argument(
        "--limit-references",
        type=int,
        default=None,
        help="Limit to first N reference PDBs (sorted) per target; useful for smoke tests.",
    )
    p.add_argument(
        "--limit-models",
        type=int,
        default=None,
        help="Limit to first N model PDBs (sorted) per target; useful for smoke tests.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _collect_pdbs(base_dir: Path, processed_dirname: str, prefer_processed: bool) -> List[Path]:
    """
    Directory contract:
      <base_dir>/*.pdb
      <base_dir>/processed/*.pdb (optional, preferred if non-empty and prefer_processed==True)
    """
    if prefer_processed:
        proc = base_dir / processed_dirname
        if proc.is_dir():
            proc_pdbs = sorted(proc.glob("*.pdb"))
            if proc_pdbs:
                return proc_pdbs
    return sorted(base_dir.glob("*.pdb"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    results_root = args.results_root if args.results_root else cfg.outputs.results_root
    workers = int(args.workers) if args.workers is not None else cfg.parallelism.max_workers
    if workers <= 0:
        raise SystemExit("--workers must be >= 1")
    if args.limit_references is not None and args.limit_references <= 0:
        raise SystemExit("--limit-references must be >= 1")
    if args.limit_models is not None and args.limit_models <= 0:
        raise SystemExit("--limit-models must be >= 1")

    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = results_root / run_id

    targets: List[Path]
    if args.target:
        targets = [p.expanduser().resolve() for p in args.target]
    else:
        targets = [p for p in _iter_target_dirs(args.targets_root.expanduser().resolve())]

    if not targets:
        print("[error] no targets found", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[dry-run] run_id={run_id}")
        print(f"[dry-run] results_root={results_root}")
        print(f"[dry-run] targets={len(targets)}")
        return 0

    # Ensure docker images exist before doing any work.
    _require_images(cfg.docker)

    residue_map = _load_two_column_map(cfg.inputs.residues_list)
    atom_map = _load_two_column_map(cfg.inputs.atoms_list)

    run_root.mkdir(parents=True, exist_ok=True)

    run_manifest: Dict[str, object] = {
        "run_id": run_id,
        "targets_root": str(args.targets_root),
        "targets": [str(t) for t in targets],
        "config": str(args.config),
        "workers": workers,
        "limits": {
            "references": int(args.limit_references) if args.limit_references is not None else None,
            "models": int(args.limit_models) if args.limit_models is not None else None,
        },
        "docker": {
            "platform": cfg.docker.platform,
            "images": {
                "tools": cfg.docker.tools_image,
                "gdt": cfg.docker.gdt_image,
                "ost": cfg.docker.ost_image,
            },
        },
        "atoms_whitelist": list(cfg.whitelist.all_atoms),
        "residues_list": str(cfg.inputs.residues_list),
        "atoms_list": str(cfg.inputs.atoms_list),
        "targets_processed": [],
        "status": "filter_and_chainmap_done",
    }

    metric_futures = []

    def submit_metric(
        ex: ThreadPoolExecutor,
        *,
        metric: str,
        pair_dir: Path,
        ref_pdb: Path,
        model_pdb: Path,
    ) -> None:
        out_dir = pair_dir / metric

        def runner() -> Dict[str, object]:
            try:
                if metric == "rmsd":
                    res = _compute_rmsd(ref_pdb=ref_pdb, model_pdb=model_pdb, whitelist_atoms=cfg.whitelist.all_atoms)
                elif metric == "inf":
                    res = _compute_inf(pair_dir=pair_dir, ref_pdb=ref_pdb, model_pdb=model_pdb, docker=cfg.docker)
                elif metric == "tm":
                    res = _compute_tm(pair_dir=pair_dir, ref_pdb=ref_pdb, model_pdb=model_pdb, docker=cfg.docker)
                elif metric == "mcq":
                    res = _compute_mcq(pair_dir=pair_dir, ref_pdb=ref_pdb, model_pdb=model_pdb, docker=cfg.docker)
                elif metric == "gdt":
                    res = _compute_gdt(pair_dir=pair_dir, ref_pdb=ref_pdb, model_pdb=model_pdb, docker=cfg.docker)
                elif metric == "lddt":
                    res = _compute_lddt(pair_dir=pair_dir, ref_pdb=ref_pdb, model_pdb=model_pdb, docker=cfg.docker)
                elif metric == "clash":
                    res = _compute_clash(pair_dir=pair_dir, ref_pdb=ref_pdb, model_pdb=model_pdb, docker=cfg.docker)
                else:
                    res = _metric_error_result(metric, "Unknown metric")
            except Exception as e:
                res = _metric_error_result(metric, f"Exception: {e}")
            _write_json(out_dir / "result.json", res)
            return res

        metric_futures.append(ex.submit(runner))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for target_dir in targets:
            target_name = target_dir.name
            out_target = run_root / target_name
            out_target.mkdir(parents=True, exist_ok=True)

            ref_dir = target_dir / cfg.inputs.references_dirname
            model_dir = target_dir / cfg.inputs.models_dirname
            if not ref_dir.is_dir() or not model_dir.is_dir():
                (out_target / "manifest.json").write_text(
                    json.dumps(
                        {
                            "target": target_name,
                            "ok": False,
                            "error": f"Missing references/ or models/ under {target_dir}",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                continue

            ref_inputs = _collect_pdbs(ref_dir, cfg.inputs.processed_dirname, cfg.inputs.prefer_processed)
            model_inputs = _collect_pdbs(model_dir, cfg.inputs.processed_dirname, cfg.inputs.prefer_processed)
            if args.limit_references is not None:
                ref_inputs = ref_inputs[: int(args.limit_references)]
            if args.limit_models is not None:
                model_inputs = model_inputs[: int(args.limit_models)]

            cache_refs = out_target / "_cache" / "references"
            cache_models = out_target / "_cache" / "models"
            cache_refs.mkdir(parents=True, exist_ok=True)
            cache_models.mkdir(parents=True, exist_ok=True)

            ref_filtered: Dict[Path, FilterResult] = {}
            model_filtered: Dict[Path, FilterResult] = {}

            # Filter references
            for ref_in in ref_inputs:
                ref_out = cache_refs / f"{ref_in.stem}.filtered.pdb"
                ref_filtered[ref_in] = _filter_pdb(
                    ref_in, ref_out, residue_map, atom_map, cfg.whitelist.all_atoms
                )

            # Filter models
            for m_in in model_inputs:
                m_out = cache_models / f"{m_in.stem}.filtered.pdb"
                model_filtered[m_in] = _filter_pdb(
                    m_in, m_out, residue_map, atom_map, cfg.whitelist.all_atoms
                )

            pairs_manifest: List[Dict[str, object]] = []
            for ref_in in ref_inputs:
                fr = ref_filtered.get(ref_in)
                if fr is None or not fr.ok:
                    continue
                for m_in in model_inputs:
                    fm = model_filtered.get(m_in)
                    if fm is None or not fm.ok:
                        continue

                    pair_dir = out_target / ref_in.stem / m_in.stem
                    inputs_dir = pair_dir / "inputs"
                    inputs_dir.mkdir(parents=True, exist_ok=True)

                    pair_ref = inputs_dir / "reference.filtered.pdb"
                    pair_model = inputs_dir / "model.filtered.mapped.pdb"

                    # Copy reference filtered file per pair (explicit & auditable inputs)
                    pair_ref.write_text(fr.output_path.read_text(encoding="utf-8"), encoding="utf-8")

                    chain_src = "auto"
                    mapping: Dict[str, str] = {}
                    map_errors: List[str] = []
                    map_warnings: List[str] = []
                    sim: Optional[List[List[float]]] = None

                    explicit = _load_explicit_chain_mapping(target_dir, cfg.inputs, ref_in, m_in)
                    if explicit is not None:
                        p, mapping = explicit
                        chain_src = f"explicit:{p}"
                    else:
                        cmr = _auto_chain_mapping(fr, fm)
                        chain_src = cmr.source
                        mapping = cmr.mapping
                        map_errors.extend(list(cmr.errors))
                        map_warnings.extend(list(cmr.warnings))
                        sim = cmr.similarity_matrix

                    ok_map, strict_errors = _validate_chain_mapping_strict(fr, fm, mapping)
                    if not ok_map:
                        map_errors.extend(strict_errors)
                    else:
                        ref_chain_order = sorted(fr.chains.keys())
                        _rewrite_model_chains(fm.output_path, pair_model, mapping, ref_chain_order)

                    pair_manifest = {
                        "reference_input": str(ref_in),
                        "model_input": str(m_in),
                        "reference_filtered": str(pair_ref),
                        "model_filtered": str(pair_model) if ok_map else None,
                        "ok": ok_map,
                        "filter": {
                            "reference": {
                                "ok": fr.ok,
                                "errors": list(fr.errors),
                                "atoms_in": fr.atoms_in,
                                "atoms_out": fr.atoms_out,
                                "residues_out": fr.residues_out,
                                "chains": {k: {"seq": v.sequence, "len": len(v.sequence)} for k, v in fr.chains.items()},
                            },
                            "model": {
                                "ok": fm.ok,
                                "errors": list(fm.errors),
                                "atoms_in": fm.atoms_in,
                                "atoms_out": fm.atoms_out,
                                "residues_out": fm.residues_out,
                                "chains": {k: {"seq": v.sequence, "len": len(v.sequence)} for k, v in fm.chains.items()},
                            },
                        },
                        "chain_mapping": {
                            "source": chain_src,
                            "mapping": mapping,
                            "ok": ok_map,
                            "errors": map_errors,
                            "warnings": map_warnings,
                            "similarity": sim,
                        },
                        "metrics": {},
                    }
                    (pair_dir / "manifest.json").write_text(
                        json.dumps(pair_manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    if ok_map:
                        for metric in ("rmsd", "inf", "tm", "mcq", "gdt", "lddt", "clash"):
                            submit_metric(
                                ex, metric=metric, pair_dir=pair_dir, ref_pdb=pair_ref, model_pdb=pair_model
                            )
                    pairs_manifest.append(
                        {
                            "ref": ref_in.name,
                            "model": m_in.name,
                            "pair_dir": str(pair_dir),
                            "ok": ok_map,
                        }
                    )

            target_manifest = {
                "target": target_name,
                "input_dir": str(target_dir),
                "references": [str(p) for p in ref_inputs],
                "models": [str(p) for p in model_inputs],
                "pairs": pairs_manifest,
                "status": "filter_and_chainmap_done",
            }
            (out_target / "manifest.json").write_text(
                json.dumps(target_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            run_manifest["targets_processed"].append(
                {"target": target_name, "out_dir": str(out_target), "pairs": len(pairs_manifest)}
            )

        for _f in as_completed(metric_futures):
            # Results are written per-metric; here we just surface exceptions.
            try:
                _f.result()
            except Exception:
                pass

    # Summarize per target
    for t in list(run_manifest.get("targets_processed", [])):
        try:
            out_dir = Path(str(t.get("out_dir", "")))
            if out_dir.is_dir():
                _summarize_target(out_dir)
                t["tables"] = {
                    "summary": str((out_dir / "tables" / "summary_all_pairs.csv")),
                    "errors": str((out_dir / "tables" / "errors.csv")),
                }
        except Exception as e:
            t["summary_error"] = str(e)

    run_manifest["status"] = "tables_done"
    _write_json(run_root / "manifest.json", run_manifest)
    print(f"[ok] finished metrics + tables: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
