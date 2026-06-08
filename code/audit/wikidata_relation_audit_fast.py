from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wikidata_relation_audit import (
    DEFAULT_LABELS,
    DEFAULT_LOCAL_EVIDENCE_DIRS,
    DEFAULT_PID_CACHE,
    DEFAULT_SUBMISSION,
    DEFAULT_TOP30,
    DEFAULT_TRAIN_DIR,
    key,
    load_local_evidence,
    load_top30,
    norm_text,
    parse_number,
    parse_year_or_date,
    values_match,
)


SEARCH_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "CTA-Wikidata-relation-audit-fast/1.0 (local competition verification)"


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32))
    return session


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def is_literal_like(text: str) -> bool:
    return parse_number(text) is not None or parse_year_or_date(text)[0] is not None


def resolve_one(name: str) -> tuple[str, str | None]:
    if not name or is_literal_like(name):
        return name.casefold(), None
    session = make_session()
    try:
        response = session.get(
            SEARCH_API,
            params={
                "action": "wbsearchentities",
                "search": name,
                "language": "en",
                "uselang": "en",
                "type": "item",
                "limit": 1,
                "format": "json",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("search"):
            return name.casefold(), data["search"][0].get("id")
    except Exception:
        return name.casefold(), None
    return name.casefold(), None


def resolve_entities(names: list[str], cache: dict[str, str | None], workers: int, out_dir: Path) -> dict[str, str | None]:
    pending = [name for name in names if name.casefold() not in cache]
    if not pending:
        return cache
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(resolve_one, name) for name in pending]
        for future in as_completed(futures):
            k, qid = future.result()
            cache[k] = qid
            done += 1
            if done % 250 == 0 or done == len(pending):
                dump_json(out_dir / "entity_qid_cache.json", cache)
                print(f"[resolve] {done}/{len(pending)} new, total_cache={len(cache)}", flush=True)
    return cache


def fetch_claim_batch(qids: list[str]) -> dict[str, dict[str, Any] | None]:
    session = make_session()
    try:
        response = session.get(
            SEARCH_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(qids),
                "props": "claims",
                "format": "json",
            },
            timeout=60,
        )
        response.raise_for_status()
        entities = response.json().get("entities", {})
        return {qid: entities.get(qid, {}).get("claims", {}) for qid in qids}
    except Exception:
        return {qid: None for qid in qids}


def fetch_claims(qids: set[str], cache: dict[str, dict[str, Any] | None], workers: int, out_dir: Path) -> dict[str, dict[str, Any] | None]:
    pending = [qid for qid in sorted(qids) if qid and qid not in cache]
    batches = [pending[i : i + 50] for i in range(0, len(pending), 50)]
    if not batches:
        return cache
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_claim_batch, batch) for batch in batches]
        for future in as_completed(futures):
            result = future.result()
            cache.update(result)
            done += len(result)
            if done % 500 == 0 or done >= len(pending):
                dump_json(out_dir / "wikidata_claim_cache.json", cache)
                print(f"[claims] {done}/{len(pending)} new, total_cache={len(cache)}", flush=True)
    return cache


def best_wiki_hit(
    subject_qid: str | None,
    object_qid: str | None,
    subject: str,
    obj: str,
    candidates: list[str],
    label_to_pid: dict[str, str],
    pid_to_label: dict[str, str],
    claims: dict[str, dict[str, Any] | None],
    allow_reverse: bool,
) -> tuple[str | None, str | None, str]:
    rank_by_label = {label: i for i, label in enumerate(candidates)}
    candidate_pids = {label_to_pid[label] for label in candidates if label in label_to_pid}

    hits: list[tuple[int, str, str, str]] = []
    if subject_qid and claims.get(subject_qid):
        for pid, claim_list in claims[subject_qid].items():
            label = pid_to_label.get(pid)
            if pid not in candidate_pids or label not in rank_by_label:
                continue
            for claim in claim_list:
                datavalue = claim.get("mainsnak", {}).get("datavalue")
                if datavalue and values_match(datavalue, obj, object_qid):
                    hits.append((rank_by_label[label], label, pid, "Corrected by Wiki direct"))
                    break

    if allow_reverse and object_qid and subject_qid and claims.get(object_qid):
        for pid, claim_list in claims[object_qid].items():
            label = pid_to_label.get(pid)
            if pid not in candidate_pids or label not in rank_by_label:
                continue
            for claim in claim_list:
                datavalue = claim.get("mainsnak", {}).get("datavalue")
                if datavalue and values_match(datavalue, subject, subject_qid):
                    hits.append((rank_by_label[label], label, pid, "Corrected by Wiki reverse"))
                    break

    if not hits:
        return None, None, "No Wiki direct edge"
    _, label, pid, status = sorted(hits)[0]
    return label, pid, status


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [line.strip() for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    label_set = set(labels)
    pid_cache = json.loads(Path(args.pid_cache).read_text(encoding="utf-8"))
    label_to_pid = {label: pid for label, pid in pid_cache.items() if label in label_set and isinstance(pid, str) and pid.startswith("P")}
    pid_to_label = {pid: label for label, pid in label_to_pid.items()}

    submission = pd.read_csv(args.submission, encoding="utf-8")
    top30 = load_top30(Path(args.top30), submission)
    local_evidence = load_local_evidence(label_set, [Path(p) for p in args.local_evidence_dirs], Path(args.train_dir))

    entity_cache_path = out_dir / "entity_qid_cache.json"
    claim_cache_path = out_dir / "wikidata_claim_cache.json"
    entity_cache: dict[str, str | None] = load_json(entity_cache_path, {})
    claim_cache: dict[str, dict[str, Any] | None] = load_json(claim_cache_path, {})

    names = sorted({norm_text(v) for col in ["Subject", "Object"] for v in submission[col].tolist() if norm_text(v)})
    print(f"[start] rows={len(submission)} labels={len(labels)} unique_names={len(names)}", flush=True)
    entity_cache = resolve_entities(names, entity_cache, args.search_workers, out_dir)
    dump_json(entity_cache_path, entity_cache)

    needed_qids = {qid for qid in entity_cache.values() if isinstance(qid, str) and qid.startswith("Q")}
    print(f"[qid] resolved={len(needed_qids)} cache_entries={len(entity_cache)}", flush=True)
    claim_cache = fetch_claims(needed_qids, claim_cache, args.claim_workers, out_dir)
    dump_json(claim_cache_path, claim_cache)

    rows: list[dict[str, Any]] = []
    labels_out: list[str] = []
    for i, row in submission.iterrows():
        subject = norm_text(row["Subject"])
        obj = norm_text(row["Object"])
        candidates = [label for label, _, _ in top30[i]]
        original = candidates[0]
        corrected = original
        status = "Unverified Keep Original"
        evidence_pid = label_to_pid.get(original, "")
        evidence_source = ""

        subject_qid = entity_cache.get(subject.casefold())
        object_qid = entity_cache.get(obj.casefold()) if not is_literal_like(obj) else None
        wiki_label, wiki_pid, wiki_status = best_wiki_hit(
            subject_qid,
            object_qid,
            subject,
            obj,
            candidates,
            label_to_pid,
            pid_to_label,
            claim_cache,
            args.allow_reverse,
        )

        if wiki_label:
            corrected = wiki_label
            status = "Wiki confirms original" if corrected == original else wiki_status
            evidence_pid = wiki_pid or ""
            evidence_source = "wikidata_wbgetentities"
        else:
            local_hits = [label for label in local_evidence.get(key(subject, obj), []) if label in candidates]
            if local_hits:
                rank_by_label = {label: idx for idx, label in enumerate(candidates)}
                corrected = sorted(local_hits, key=lambda label: rank_by_label[label])[0]
                status = "Local evidence confirms original" if corrected == original else "Corrected by local graph evidence"
                evidence_pid = label_to_pid.get(corrected, "")
                evidence_source = "local_csv"

        if corrected not in label_set:
            corrected = original
            status = "Fallback Keep Original"
            evidence_pid = label_to_pid.get(original, "")
            evidence_source = ""

        labels_out.append(corrected)
        rows.append(
            {
                "RowID": i,
                "Subject": subject,
                "Object": obj,
                "Original_Label": original,
                "Corrected_Label": corrected,
                "Status": status,
                "Evidence_PID": evidence_pid,
                "Evidence_Source": evidence_source,
                "Subject_QID": subject_qid or "",
                "Object_QID": object_qid or "",
                "Top30": " | ".join(candidates),
            }
        )

    final = pd.DataFrame(
        {
            "Subject": submission["Subject"].map(norm_text),
            "Object": submission["Object"].map(norm_text),
            "Label": labels_out,
        }
    )
    audit = pd.DataFrame(rows)
    final.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out_dir / "audit_report.csv", index=False, encoding="utf-8-sig")
    summary = {
        "rows": len(final),
        "labels": len(labels),
        "pid_labels": len(label_to_pid),
        "status_counts": audit["Status"].value_counts().to_dict(),
        "changed": int((audit["Original_Label"] != audit["Corrected_Label"]).sum()),
        "resolved_qids": len(needed_qids),
        "claim_cache": len(claim_cache),
        "time": datetime.now().isoformat(timespec="seconds"),
        "final_csv": str(out_dir / "submission.csv"),
        "audit_csv": str(out_dir / "audit_report.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--pid-cache", default=str(DEFAULT_PID_CACHE))
    parser.add_argument("--top30", default=str(DEFAULT_TOP30))
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--local-evidence-dirs", nargs="*", default=[str(p) for p in DEFAULT_LOCAL_EVIDENCE_DIRS])
    parser.add_argument("--out-dir", default=str(Path.cwd() / "wikidata_audit_output_online_fast"))
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--search-workers", type=int, default=10)
    parser.add_argument("--claim-workers", type=int, default=6)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
