from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wikidata_relation_audit import DEFAULT_PID_CACHE, DEFAULT_TOP30, load_top30, norm_text, parse_number, parse_year_or_date
from wikidata_relation_audit_titleclaims import (
    DEFAULT_LABELS,
    DEFAULT_LOCAL_EVIDENCE_DIRS,
    DEFAULT_SUBMISSION,
    DEFAULT_TRAIN_DIR,
    USER_AGENT,
    dump_json,
    fetch_labels,
    find_hit,
    load_json,
)


API = "https://www.wikidata.org/w/api.php"


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=12, pool_maxsize=12))
    return session


def is_disambig(entity: dict[str, Any] | None) -> bool:
    if not entity:
        return False
    for claim in entity.get("claims", {}).get("P31", []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and value.get("id") == "Q4167410":
            return True
    return False


def search_one(name: str) -> tuple[str, str | None]:
    session = make_session()
    try:
        response = session.get(
            API,
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
        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", "30")))
            response = session.get(
                API,
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


def resolve_search(names: list[str], cache: dict[str, str | None], out_dir: Path, workers: int) -> dict[str, str | None]:
    pending = [name for name in names if name.casefold() not in cache]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(search_one, name) for name in pending]
        for future in as_completed(futures):
            k, qid = future.result()
            cache[k] = qid
            done += 1
            if done % 100 == 0 or done == len(pending):
                dump_json(out_dir / "search_qid_cache.json", cache)
                print(f"[search] {done}/{len(pending)}", flush=True)
    return cache


def fetch_claim_batch(qids: list[str]) -> dict[str, dict[str, Any] | None]:
    session = make_session()
    try:
        response = session.get(
            API,
            params={"action": "wbgetentities", "ids": "|".join(qids), "props": "claims", "format": "json"},
            timeout=75,
        )
        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", "60")))
            response = session.get(
                API,
                params={"action": "wbgetentities", "ids": "|".join(qids), "props": "claims", "format": "json"},
                timeout=75,
            )
        response.raise_for_status()
        entities = response.json().get("entities", {})
        return {qid: entities.get(qid, {}).get("claims", {}) for qid in qids}
    except Exception:
        return {qid: None for qid in qids}


def fetch_claims(qids: set[str], cache: dict[str, dict[str, Any] | None], out_dir: Path) -> dict[str, dict[str, Any] | None]:
    pending = [qid for qid in sorted(qids, key=lambda q: int(q[1:]) if q[1:].isdigit() else 10**18) if qid and qid not in cache]
    done = 0
    for i in range(0, len(pending), 50):
        result = fetch_claim_batch(pending[i : i + 50])
        cache.update(result)
        done += len(result)
        if done % 500 == 0 or done == len(pending):
            dump_json(out_dir / "search_claim_cache.json", cache)
        print(f"[claims] {done}/{len(pending)}", flush=True)
        time.sleep(0.8)
    return cache


def run(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [line.strip() for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    label_set = set(labels)
    pid_cache = json.loads(Path(args.pid_cache).read_text(encoding="utf-8"))
    label_to_pid = {label: pid for label, pid in pid_cache.items() if label in label_set and isinstance(pid, str) and pid.startswith("P")}

    submission = pd.read_csv(args.submission, encoding="utf-8")
    top30 = load_top30(Path(args.top30), submission)
    audit = pd.read_csv(base_dir / "audit_report.csv", encoding="utf-8-sig")
    title_cache = load_json(base_dir / "title_entity_claim_cache.json", {})
    qid_label_cache = load_json(base_dir / "qid_label_cache.json", {})

    target_names: set[str] = set()
    for _, row in audit[audit["Status"].eq("Unverified Keep Original")].iterrows():
        subject = norm_text(row["Subject"])
        entity = title_cache.get(subject.casefold())
        if not entity or is_disambig(entity):
            target_names.add(subject)

    search_cache = load_json(out_dir / "search_qid_cache.json", {})
    search_cache = resolve_search(sorted(target_names), search_cache, out_dir, args.search_workers)
    dump_json(out_dir / "search_qid_cache.json", search_cache)

    qids = {qid for qid in search_cache.values() if isinstance(qid, str) and qid.startswith("Q")}
    search_claim_cache = load_json(out_dir / "search_claim_cache.json", {})
    search_claim_cache = fetch_claims(qids, search_claim_cache, out_dir)
    dump_json(out_dir / "search_claim_cache.json", search_claim_cache)

    needed_value_qids: set[str] = set()
    for i, row in submission.iterrows():
        if audit.loc[i, "Status"] != "Unverified Keep Original":
            continue
        subject_qid = search_cache.get(norm_text(row["Subject"]).casefold())
        claims = search_claim_cache.get(subject_qid) or {}
        candidates = [label for label, _, _ in top30[i]]
        for label in candidates:
            pid = label_to_pid.get(label)
            for claim in claims.get(pid, []):
                value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(value, dict) and value.get("entity-type") == "item":
                    needed_value_qids.add(f"Q{value.get('numeric-id')}")

    qid_label_cache = fetch_labels(needed_value_qids, qid_label_cache, out_dir, 50, 0.8)
    dump_json(out_dir / "qid_label_cache.json", qid_label_cache)

    refined_cache: dict[str, dict[str, Any] | None] = {}
    for name, qid in search_cache.items():
        if qid and qid in search_claim_cache and search_claim_cache[qid] is not None:
            refined_cache[name] = {"id": qid, "claims": search_claim_cache[qid]}

    changed = 0
    for i, row in submission.iterrows():
        if audit.loc[i, "Status"] != "Unverified Keep Original":
            continue
        subject = norm_text(row["Subject"])
        obj = norm_text(row["Object"])
        candidates = [label for label, _, _ in top30[i]]
        label, pid = find_hit(subject, obj, candidates, label_to_pid, refined_cache, qid_label_cache)
        if label:
            audit.loc[i, "Corrected_Label"] = label
            audit.loc[i, "Status"] = "Wiki confirms original" if label == audit.loc[i, "Original_Label"] else "Corrected by Wiki search direct"
            audit.loc[i, "Evidence_PID"] = pid or ""
            audit.loc[i, "Evidence_Source"] = "wikidata_search_claims"
            audit.loc[i, "Subject_QID"] = refined_cache[subject.casefold()].get("id", "")
            changed += int(label != audit.loc[i, "Original_Label"])

    final = pd.DataFrame(
        {
            "Subject": submission["Subject"].map(norm_text),
            "Object": submission["Object"].map(norm_text),
            "Label": audit["Corrected_Label"].tolist(),
        }
    )
    final.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out_dir / "audit_report.csv", index=False, encoding="utf-8-sig")
    summary = {
        "rows": len(final),
        "status_counts": audit["Status"].value_counts().to_dict(),
        "changed_total": int((audit["Original_Label"] != audit["Corrected_Label"]).sum()),
        "changed_by_refine": changed,
        "searched_names": len(target_names),
        "search_qids": len(qids),
        "time": pd.Timestamp.now().isoformat(),
        "final_csv": str(out_dir / "submission.csv"),
        "audit_csv": str(out_dir / "audit_report.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(Path.cwd() / "wikidata_audit_output_titleclaims"))
    parser.add_argument("--out-dir", default=str(Path.cwd() / "wikidata_audit_output_refined"))
    parser.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--pid-cache", default=str(DEFAULT_PID_CACHE))
    parser.add_argument("--top30", default=str(DEFAULT_TOP30))
    parser.add_argument("--search-workers", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
