#!/usr/bin/env python3
"""Derive a source-bound revision of the finite Triton B1 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_hash(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--add-source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    parent = json.loads(args.parent.read_text())
    parent_payload = {key: value for key, value in parent.items() if key != "contract_id"}
    if canonical_hash(parent_payload) != parent.get("contract_id"):
        raise ValueError("parent contract ID integrity check failed")
    payload = dict(parent_payload)
    sources = dict(payload.get("source_sha256", {}))
    added = {}
    for source_arg in args.add_source:
        source = source_arg.as_posix()
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source in sources and sources[source] != digest:
            raise ValueError(f"source already pinned with a different digest: {source}")
        sources[source] = digest
        added[source] = digest
    payload["source_sha256"] = dict(sorted(sources.items()))
    provenance = dict(payload.get("provenance", {}))
    provenance["derived_from_contract"] = str(args.parent)
    provenance["added_source_sha256"] = added
    payload["provenance"] = provenance
    contract = {**payload, "contract_id": canonical_hash(payload)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    print(contract["contract_id"])


if __name__ == "__main__":
    main()
