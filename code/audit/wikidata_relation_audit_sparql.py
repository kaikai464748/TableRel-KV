from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
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
)


SPARQL_API = "https://query.wikidata.org/sparql"
USER_AGENT = "CTA-Wikidata-relation-audit-sparql/1.0 (local competition verification)"


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    return session


def sparql_string(text: str) -> str:
    return json.dumps(norm_text(text), ensure_ascii=False)


def row_values(rows: list[tuple[int, str, str]]) -> str:
    return "\n".join(f"({idx} {sparql_string(subject)} {sparql_string(obj)})" for idx, subject, obj in rows)


def pid_values(pids: set[str]) -> str:
    return " ".join(f"wdt:{pid}" for pid in sorted(pids))


def query_sparql(session: requests.Session, query: str, retries: int = 4) -> list[dict[str, Any]]:
    for attempt in range(retries):
        try:
            response = session.get(SPARQL_API, params={"query": query, "format": "json"}, timeout=90)
            if response.status_code == 429:
                time.sleep(int(response.headers.get("Retry-After", "30")))
                continue
            response.raise_for_status()
            return response.json().get("results", {}).get("bindings", [])
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[sparql-error] {exc}", flush=True)
                return []
            time.sleep(2 * (attempt + 1))
    return []


def literal_matches(value: str, target: str) -> bool:
    v = norm_text(value)
    t = norm_text(target)
    if v == t:
        return True
    vn = parse_number(v)
    tn = parse_number(t)
    if vn is not None and tn is not None and math.isclose(vn, tn, rel_tol=1e-9, abs_tol=1e-6):
        return True
    vy, vd = parse_year_or_date(v.replace("T00:00:00Z", ""))
    ty, td = parse_year_or_date(t)
    if vy is not None and ty is not None:
        if vd and td:
            return vd == td
        return vy == ty
    return False


def build_entity_query(rows: list[tuple[int, str, str]], pids: set[str], reverse: bool = False) -> str:
    if reverse:
        triple = """
          ?objectItem rdfs:label ?objectLabel .
          FILTER(LANG(?objectLabel) = "en" && STR(?objectLabel) = ?objName)
          ?subjectItem rdfs:label ?subjectLabel .
          FILTER(LANG(?subjectLabel) = "en" && STR(?subjectLabel) = ?subjName)
          ?objectItem ?p ?subjectItem .
        """
    else:
        triple = """
          ?subjectItem rdfs:label ?subjectLabel .
          FILTER(LANG(?subjectLabel) = "en" && STR(?subjectLabel) = ?subjName)
          ?objectItem rdfs:label ?objectLabel .
          FILTER(LANG(?objectLabel) = "en" && STR(?objectLabel) = ?objName)
          ?subjectItem ?p ?objectItem .
        """
    return f"""
    SELECT DISTINCT ?row ?pid WHERE {{
      VALUES (?row ?subjName ?objName) {{
        {row_values(rows)}
      }}
      VALUES ?p {{ {pid_values(pids)} }}
      {triple}
      BIND(STRAFTER(STR(?p), "http://www.wikidata.org/prop/direct/") AS ?pid)
    }}
    """


def build_literal_query(rows: list[tuple[int, str, str]], pids: set[str]) -> str:
    return f"""
    SELECT DISTINCT ?row ?pid ?value WHERE {{
      VALUES (?row ?subjName ?objName) {{
        {row_values(rows)}
      }}
      VALUES ?p {{ {pid_values(pids)} }}
      ?subjectItem rdfs:label ?subjectLabel .
      FILTER(LANG(?subjectLabel) = "en" && STR(?subjectLabel) = ?subjName)
      ?subjectItem ?p ?value .
      FILTER(!isIRI(?value))
      BIND(STRAFTER(STR(?p), "http://www.wikidata.org/prop/direct/") AS ?pid)
    }}
    """


def collect_sparql_hits(
    session: requests.Session,
    rows: list[tuple[int, str, str]],
    row_candidates: dict[int, list[str]],
    label_to_pid: dict[str, str],
    pid_to_label: dict[str, str],
    allow_reverse: bool,
) -> dict[int, list[tuple[str, str, str]]]:
    batch_pids = {label_to_pid[label] for idx, _, _ in rows for label in row_candidates[idx] if label in label_to_pid}
    hits: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    if not batch_pids:
        return hits

    for binding in query_sparql(session, build_entity_query(rows, batch_pids, reverse=False)):
        idx = int(binding["row"]["value"])
        pid = binding["pid"]["value"]
        label = pid_to_label.get(pid)
        if label in row_candidates[idx]:
            hits[idx].append((label, pid, "Corrected by Wiki direct"))

    if allow_reverse:
        for binding in query_sparql(session, build_entity_query(rows, batch_pids, reverse=True)):
            idx = int(binding["row"]["value"])
            pid = binding["pid"]["value"]
            label = pid_to_label.get(pid)
            if label in row_candidates[idx]:
                hits[idx].append((label, pid, "Corrected by Wiki reverse"))

    literal_targets = {idx: obj for idx, _, obj in rows}
    for binding in query_sparql(session, build_literal_query(rows, batch_pids)):
        idx = int(binding["row"]["value"])
        pid = binding["pid"]["value"]
        label = pid_to_label.get(pid)
        value = binding["value"]["value"]
        if label in row_candidates[idx] and literal_matches(value, literal_targets[idx]):
            hits[idx].append((label, pid, "Corrected by Wiki literal"))

    return hits


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
    row_candidates = {i: [label for label, _, _ in top30[i]] for i in range(len(submission))}
    local_evidence = load_local_evidence(label_set, [Path(p) for p in args.local_evidence_dirs], Path(args.train_dir))

    session = make_session()
    all_hits: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    total = len(submission)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_rows = [
            (i, norm_text(submission.loc[i, "Subject"]), norm_text(submission.loc[i, "Object"]))
            for i in range(start, end)
        ]
        batch_hits = collect_sparql_hits(session, batch_rows, row_candidates, label_to_pid, pid_to_label, args.allow_reverse)
        for idx, hits in batch_hits.items():
            all_hits[idx].extend(hits)
        if (end % args.save_every == 0) or end == total:
            (out_dir / "sparql_hits_checkpoint.json").write_text(
                json.dumps({str(k): v for k, v in all_hits.items()}, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"[sparql] {end}/{total} hit_rows={len(all_hits)}", flush=True)
        time.sleep(args.delay)

    rows: list[dict[str, Any]] = []
    labels_out: list[str] = []
    for i, row in submission.iterrows():
        subject = norm_text(row["Subject"])
        obj = norm_text(row["Object"])
        candidates = row_candidates[i]
        rank = {label: pos for pos, label in enumerate(candidates)}
        original = candidates[0]
        corrected = original
        status = "Unverified Keep Original"
        evidence_pid = label_to_pid.get(original, "")
        evidence_source = ""

        if all_hits.get(i):
            label, pid, hit_status = sorted(all_hits[i], key=lambda item: rank.get(item[0], 999))[0]
            corrected = label
            status = "Wiki confirms original" if corrected == original else hit_status
            evidence_pid = pid
            evidence_source = "wikidata_sparql"
        else:
            local_hits = [label for label in local_evidence.get(key(subject, obj), []) if label in candidates]
            if local_hits:
                corrected = sorted(local_hits, key=lambda label: rank.get(label, 999))[0]
                status = "Local evidence confirms original" if corrected == original else "Corrected by local graph evidence"
                evidence_pid = label_to_pid.get(corrected, "")
                evidence_source = "local_csv"

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
                "Top30": " | ".join(candidates),
            }
        )

    final = pd.DataFrame({"Subject": submission["Subject"].map(norm_text), "Object": submission["Object"].map(norm_text), "Label": labels_out})
    audit = pd.DataFrame(rows)
    final.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out_dir / "audit_report.csv", index=False, encoding="utf-8-sig")
    summary = {
        "rows": len(final),
        "labels": len(labels),
        "pid_labels": len(label_to_pid),
        "status_counts": audit["Status"].value_counts().to_dict(),
        "changed": int((audit["Original_Label"] != audit["Corrected_Label"]).sum()),
        "sparql_hit_rows": len(all_hits),
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
    parser.add_argument("--out-dir", default=str(Path.cwd() / "wikidata_audit_output_sparql"))
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
