"""I/O helpers for .psyscan simulation files — Polars-based CSV export."""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


def _parse_pred_resp(pred_resp: Any) -> tuple[str, dict]:
    """Return (raw_content_str, {field: value}) from any pred_resp shape.

    Handles three cases:
    - LangChain AIMessage (or any object with .content): extract .content
    - dict with "raw"/"parsed" keys (RawPredMsgModel): extract parsed content
    - plain dict with "content" key: extract directly
    """
    if pred_resp is None:
        return "", {}

    # LangChain AIMessage and similar objects expose .content
    if hasattr(pred_resp, "content"):
        raw = pred_resp.content
        if not isinstance(raw, str):
            raw = str(raw)
    elif isinstance(pred_resp, dict):
        # RawPredMsgModel shape: {"raw": ..., "parsed": ...}
        if "raw" in pred_resp and "parsed" in pred_resp:
            inner = pred_resp["parsed"]
            raw = inner.get("content", str(inner)) if isinstance(inner, dict) else str(inner)
        elif "content" in pred_resp:
            raw = pred_resp["content"]
        else:
            raw = json.dumps(pred_resp, default=str)
    else:
        return str(pred_resp), {}

    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
    return raw, (parsed if isinstance(parsed, dict) else {})


def _parse_tunnel_id(tunnel_id: str) -> dict[str, str]:
    """Parse model/family/memory/projectname from tunnel_id.

    Format: sidx-[N]-model-{model}-family-{family}-memory-{memory}-{projectname}
    Model names may contain '-' (e.g. smollm2:360m-instruct-fp16).
    """
    out: dict[str, str] = {}
    if not tunnel_id:
        return out
    try:
        # strip leading 'sidx-[N]-model-'
        body = re.sub(r"^sidx-\[\d+\]-model-", "", tunnel_id)
        # split from right on '-memory-' first
        left, right = body.rsplit("-memory-", 1)
        # split left from right on '-family-'
        model_part, family_part = left.rsplit("-family-", 1)
        # right = memory-projectname or just memory
        memory_part, *proj_parts = right.split("-", 1)
        out = {
            "model": model_part,
            "family": family_part,
            "memory": memory_part,
            "projectname": proj_parts[0] if proj_parts else "",
        }
    except Exception:
        pass
    return out


def _stimulus_str(stimulus: Any) -> str:
    if stimulus is None:
        return ""
    if isinstance(stimulus, list):
        # Multimodal content blocks: keep type/mime_type for auditability,
        # drop base64/url payloads so a single stimulus doesn't bloat every row.
        # A content list may legitimately mix plain strings with block dicts.
        return json.dumps(
            [
                {k: b[k] for k in ("type", "mime_type") if k in b}
                if isinstance(b, dict) else b
                for b in stimulus
            ],
            ensure_ascii=False,
        )
    if isinstance(stimulus, dict):
        return json.dumps(stimulus, ensure_ascii=False)
    return str(stimulus)


def _expcard_meta(expcard: Any) -> dict[str, Any]:
    """Extract condition-identification metadata from an ExpCard."""
    try:
        ci = expcard.card_in
    except AttributeError:
        return {}

    parser_val = getattr(ci, "parser", None)
    if parser_val is None:
        parser_str = "none"
    elif isinstance(parser_val, str):
        parser_str = parser_val
    elif isinstance(parser_val, type):
        parser_str = parser_val.__name__
    elif callable(parser_val):
        parser_str = getattr(parser_val, "__qualname__", str(parser_val))
    else:
        parser_str = str(parser_val)

    taskname = None
    if hasattr(expcard, "task_data") and isinstance(expcard.task_data, dict):
        taskname = expcard.task_data.get("taskname")

    return {
        "model": getattr(ci, "model", None),
        "family": getattr(ci, "family", None),
        "memory": getattr(ci, "memory", None),
        "memory_k": getattr(ci, "memory_k", None),
        "projectname": getattr(ci, "projectname", None),
        "cogtype": getattr(ci, "cogtype", None),
        "parser": parser_str,
        "taskname": taskname,
    }


def _trials_from_psyscan(path: Path) -> list[dict]:
    """Load trial dicts from a .psyscan file."""
    raw = json.loads(path.read_text())
    if "taskdata" in raw:
        return raw["taskdata"]
    if "simdata" in raw:
        trials: list[dict] = []
        for task in raw["simdata"]:
            trials.extend(task.get("taskdata", []))
        return trials
    return []


def _trials_to_rows(trials: list[dict], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a list of trial dicts to flat row dicts ready for DataFrame."""
    rows: list[dict[str, Any]] = []
    for t in trials:
        raw_content, resp_fields = _parse_pred_resp(t.get("pred_resp"))
        tunnel_id = t.get("tunnel_id", "")
        tunnel_info = _parse_tunnel_id(tunnel_id)

        row: dict[str, Any] = {
            "sim_idx": t.get("system_message_idx"),
            "model": meta.get("model") or tunnel_info.get("model"),
            "family": meta.get("family") or tunnel_info.get("family"),
            "memory": meta.get("memory") or tunnel_info.get("memory"),
            "memory_k": meta.get("memory_k"),
            "projectname": meta.get("projectname") or tunnel_info.get("projectname"),
            "cogtype": meta.get("cogtype"),
            "parser": meta.get("parser"),
            "taskname": t.get("taskname"),
            "tasktype": t.get("tasktype"),
            "chain_type": t.get("chain_type"),
            "trace_id": t.get("trace_id"),
            "tunnel_id": tunnel_id,
            "trial_idx": t.get("trial_idx"),
            "trcode": t.get("trcode"),
            "context_item": t.get("context_item"),
            "stimulus": _stimulus_str(t.get("stimulus")),
            "pred_resp_raw": raw_content,
            # Union-type parsers (e.g. AllResponseRMEI) nest their payload under
            # one key, e.g. {"response": {"Judgment": "external"}} -- CSV/polars
            # can't hold a nested value in a cell, so flatten it to a JSON string.
            **{
                f"resp_{k}": json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
                for k, v in resp_fields.items()
            },
            "fb_response": t.get("fb_response"),
        }
        rows.append(row)
    return rows


def _rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    """Build a Polars DataFrame from flat row dicts, coercing numeric resp_ columns."""
    if not rows:
        return pl.DataFrame()
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    col_data = {k: [row.get(k) for row in rows] for k in all_keys}
    df = pl.DataFrame(col_data, infer_schema_length=len(rows))
    for col in df.columns:
        if col.startswith("resp_") and df[col].dtype == pl.Utf8:
            try:
                cast_col = pl.col(col).cast(pl.Float64, strict=False)
                candidate = df.with_columns(cast_col)
                # only keep the cast if it doesn't introduce new nulls
                orig_nulls = df[col].null_count()
                new_nulls = candidate[col].null_count()
                if new_nulls <= orig_nulls:
                    df = candidate
            except Exception:
                pass
    return df


def _auto_path(meta: dict[str, Any], base_dir: Path | None) -> Path:
    """Generate an automatic output path from metadata."""
    parts = [
        meta.get("projectname") or "sim",
        meta.get("taskname") or "task",
        (meta.get("model") or "").replace("/", "-").replace(":", "-"),
        meta.get("memory") or "",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ]
    stem = "_".join(p for p in parts if p)
    return (base_dir or Path.cwd()) / f"{stem}.csv"


def _source_to_df(source: Any, expcard: Any = None) -> tuple[pl.DataFrame, dict[str, Any], Path | None]:
    """Collect rows from *source* and return ``(DataFrame, meta, base_dir)``.

    Does not write to disk — shared by ``to_csv`` and ``concat_csv``.
    """
    from psychscanner.scanner_models.scanner_model import ScannerModel
    from psychscanner.staging.scanner_cards import ExpCard

    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    base_dir: Path | None = None

    if isinstance(source, pl.DataFrame):
        # already a DataFrame (e.g. returned by a previous to_csv call)
        return source, meta, base_dir

    elif isinstance(source, ScannerModel):
        meta = _expcard_meta(source.expcard)
        base_dir = getattr(source, "data_root_dir", None)
        sim_data = getattr(source, "current_scanner_data", None) or []
        for participant in sim_data:
            rows.extend(_trials_to_rows(participant, meta))

    elif isinstance(source, list):
        if expcard is not None:
            meta = _expcard_meta(expcard)
            base_dir = getattr(expcard, "data_root_dir", None)
        for participant in source:
            if isinstance(participant, list):
                rows.extend(_trials_to_rows(participant, meta))
            elif isinstance(participant, dict):
                rows.extend(_trials_to_rows([participant], meta))

    elif isinstance(source, ExpCard):
        meta = _expcard_meta(source)
        base_dir = getattr(source, "data_root_dir", None)
        if base_dir and Path(base_dir).exists():
            for f in sorted(Path(base_dir).rglob("*.psyscan")):
                rows.extend(_trials_to_rows(_trials_from_psyscan(f), meta))

    elif isinstance(source, (str, Path)):
        data_dir = Path(source)
        base_dir = data_dir
        for f in sorted(data_dir.rglob("*.psyscan")):
            trials = _trials_from_psyscan(f)
            if trials:
                if not meta:
                    meta.update(_parse_tunnel_id(trials[0].get("tunnel_id", "")))
                rows.extend(_trials_to_rows(trials, meta))

    else:
        raise TypeError(
            f"source must be ScannerModel, list, ExpCard, DataFrame, or directory path; "
            f"got {type(source).__name__}"
        )

    return _rows_to_frame(rows), meta, base_dir


def to_csv(
    source: Any,
    path: str | Path | None = None,
    *,
    expcard: Any = None,
    sep: str = ",",
    combined: bool = False,
) -> pl.DataFrame:
    """Save .psyscan simulation data to CSV using Polars and return the DataFrame.

    Parameters
    ----------
    source:
        One of:

        - ``ScannerModel`` instance (after ``.run()``)
        - ``list`` returned by ``ScannerModel.run()``
        - ``ExpCard`` instance (scans ``data_root_dir`` for all ``.psyscan`` files)
        - ``str`` or ``Path`` directory (scans recursively for ``.psyscan`` files)
        - ``list`` of any of the above (when *combined* is ``True``)
    path:
        CSV output path.  Auto-generated from metadata if omitted.
    expcard:
        Optional ``ExpCard`` supplying metadata when *source* is a raw ``list``.
    sep:
        CSV field separator (default ``','``).
    combined:
        When ``True``, treat *source* as a list of sources and concatenate all
        of them into a single CSV.  Equivalent to calling ``concat_csv(source)``.

    Returns
    -------
    pl.DataFrame
        The exported data frame (empty if no trials were found).
    """
    if combined:
        return concat_csv(source, path=path, sep=sep)

    df, meta, base_dir = _source_to_df(source, expcard)

    if not df.is_empty():
        taskname = df["taskname"][0] if "taskname" in df.columns else None
        if path is None:
            out = _auto_path({**meta, "taskname": taskname}, base_dir)
        else:
            out = Path(path)
            if out.is_dir() or out.suffix == "":
                out = _auto_path({**meta, "taskname": taskname}, out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(out, separator=sep)
        print(f"Saved {len(df)} rows → {out}")
    else:
        print("No trial data found; CSV not written.")

    return df


def concat_csv(
    sources: list[Any],
    path: str | Path | None = None,
    *,
    sep: str = ",",
) -> pl.DataFrame:
    """Concatenate simulation data from multiple sources into a single CSV.

    Each element of *sources* can be anything accepted by ``to_csv``:
    a ``ScannerModel``, a ``list`` from ``.run()``, an ``ExpCard``, or a
    directory path.  All sources are read into memory, their DataFrames are
    aligned on a common column set (missing columns filled with ``null``),
    and the result is written as one CSV.

    Parameters
    ----------
    sources:
        List of sources to concatenate.  Must not be empty.
    path:
        Output CSV path.  Auto-generated in the current directory if omitted.
    sep:
        CSV field separator (default ``','``).

    Returns
    -------
    pl.DataFrame
        The concatenated data frame (empty if no trials were found in any source).
    """
    # accept a bare single source for convenience
    if not isinstance(sources, list):
        sources = [sources]
    if not sources:
        raise ValueError("sources must be a non-empty list")

    frames: list[pl.DataFrame] = []
    first_meta: dict[str, Any] = {}
    first_base: Path | None = None

    for i, src in enumerate(sources):
        df, meta, base_dir = _source_to_df(src)
        if not df.is_empty():
            frames.append(df)
        if i == 0:
            first_meta, first_base = meta, base_dir

    if not frames:
        print("No trial data found in any source; CSV not written.")
        return pl.DataFrame()

    # align schemas: collect all column names in encounter order
    all_cols = list(dict.fromkeys(col for df in frames for col in df.columns))

    # determine target dtype for each column: prefer non-Null types
    col_dtype: dict[str, Any] = {}
    for df in frames:
        for col in df.columns:
            dtype = df[col].dtype
            if dtype != pl.Null:
                col_dtype[col] = dtype

    aligned = []
    for df in frames:
        # add any missing columns as nulls cast to the known target dtype
        additions = []
        for c in all_cols:
            if c not in df.columns:
                target = col_dtype.get(c, pl.Utf8)
                additions.append(pl.lit(None).cast(target).alias(c))
        if additions:
            df = df.with_columns(additions)
        # cast Null-typed columns to their target dtype so concat doesn't fail
        casts = []
        for c in all_cols:
            if df[c].dtype == pl.Null and c in col_dtype:
                casts.append(pl.col(c).cast(col_dtype[c]))
        if casts:
            df = df.with_columns(casts)
        aligned.append(df.select(all_cols))

    combined = pl.concat(aligned, rechunk=True)

    taskname = combined["taskname"][0] if "taskname" in combined.columns else None
    if path is None:
        out = _auto_path({**first_meta, "taskname": taskname}, first_base)
    else:
        out = Path(path)
        if out.is_dir() or out.suffix == "":
            out = _auto_path({**first_meta, "taskname": taskname}, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(out, separator=sep)
    print(f"Saved {len(combined)} rows from {len(frames)} source(s) → {out}")

    return combined
