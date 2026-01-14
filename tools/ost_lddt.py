#!/usr/bin/env python3
"""
OpenStructure lDDT runner (no stereo-chemistry check).

This script is intended to be executed inside the OpenStructure docker image
(ENTRYPOINT is typically `ost`), for example:

  docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work openstructure:2.11.1-amd64 \
    ost_lddt.py model.pdb ref.pdb chain_mapping.json out.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

from ost import io
from ost.mol.alg.lddt import lDDTScorer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute global and per-residue lDDT using OpenStructure.")
    p.add_argument("model_pdb", help="Model PDB (prediction).")
    p.add_argument("ref_pdb", help="Reference PDB.")
    p.add_argument("chain_mapping_json", help="JSON mapping: {model_chain: ref_chain}.")
    p.add_argument("out_json", help="Output JSON path.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    chain_mapping: Dict[str, str] = json.load(open(args.chain_mapping_json, "r"))
    mdl = io.LoadPDB(args.model_pdb)
    ref = io.LoadPDB(args.ref_pdb)

    scorer = lDDTScorer(ref)
    lddt_score, _ = scorer.lDDT(mdl, local_lddt_prop="lddt", chain_mapping=chain_mapping)

    local_scores = {}
    for r in mdl.residues:
        key = r.GetNumber().GetNum()
        if r.HasProp("lddt"):
            local_scores[key] = r.GetFloatProp("lddt")
        else:
            local_scores[key] = None

    out = {
        "model_pdb": args.model_pdb,
        "ref_pdb": args.ref_pdb,
        "chain_mapping": chain_mapping,
        "lDDT": float(lddt_score),
        "local_lDDT": local_scores,
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

