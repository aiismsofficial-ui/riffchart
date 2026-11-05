"""Phase 1 pipeline entry points.

    python -m autochart.data.cli audit   [--limit N]   # per-song audit rows (resumable)
    python -m autochart.data.cli dedup
    python -m autochart.data.cli split
    python -m autochart.data.cli report
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from .scan import scan_all


def load_cfg(path: str) -> dict:
    from ..paths import load_config
    return load_config(path)


def cmd_audit(cfg: dict, limit: int | None) -> None:
    from .audit import audit_worker

    out_dir = Path(cfg["data_root"]) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    tagged = scan_all(cfg)
    if limit:
        tagged = tagged[:limit]
    library = {s.song_id: tag for tag, s in tagged}
    jobs = [(s, cfg, out_dir) for _, s in tagged]
    rows = []
    with ProcessPoolExecutor(cfg["audit"]["workers"]) as ex:
        futs = [ex.submit(audit_worker, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="audit", mininterval=5):
            rows.append(f.result())
    df = pd.DataFrame(rows)
    df["library"] = df.song_id.map(library)
    for col in ("audio_files", "window_lags_ms"):
        if col in df:
            df[col] = df[col].apply(lambda v: json.dumps(v) if isinstance(v, list) else v)
    df.to_parquet(out_dir / "audit_raw.parquet")
    print(df["status"].value_counts().to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "dedup", "split", "report"])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="split: re-draw the frozen split")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    if args.cmd == "audit":
        cmd_audit(cfg, args.limit)
    elif args.cmd == "dedup":
        from .dedup import run_dedup
        run_dedup(cfg)
    elif args.cmd == "split":
        from .split import run_split
        run_split(cfg, force=args.force)
    elif args.cmd == "report":
        from .report import run_report
        run_report(cfg)


if __name__ == "__main__":
    main()
