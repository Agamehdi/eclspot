"""Parser, conversion and Spotfire exports for Eclipse ASCII .RSM summaries.

The library has no Streamlit dependency so it can be tested from a terminal.
Note: Eclipse RSM is a printed tabular summary, not an EGRID/UNSMRY binary reader.
"""
from __future__ import annotations

import io
import re
import csv
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

STB_PER_M3 = 6.28981077043
SCF_PER_M3 = 35.3146667215
PSI_PER_BAR = 14.5037738007
BTU_PER_KJ = 0.947817120313

NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$")
SCALE = re.compile(r"\*\s*10\s*\*\*\s*([+-]?\d+)", re.I)


def _cells(line: str) -> list[str]:
    """Preserve positional empty cells (critical for per-column scale headers)."""
    parts = [p.strip() for p in line.rstrip("\r\n").split("\t")]
    if parts and parts[0] == "":
        parts = parts[1:]
    return parts


def _num(s: str) -> float:
    s = s.strip()
    if not s:
        return np.nan
    try:
        return float(s.replace("D", "E").replace("d", "e"))
    except ValueError:
        return np.nan


def _is_number(s: str) -> bool:
    return bool(NUM.match(s.strip()))


def _scope(keyword: str) -> str:
    # 'W' => well vectors; 'G' => group; 'F'/other => field/scalar.
    if keyword.startswith("W"):
        return "Well"
    if keyword.startswith("G"):
        return "Group"
    return "Field"


def _gas(keyword: str) -> bool:
    # FGPR, WGPR, GGPR, FGPT, WGPT, FGIP, FGIR, WGIT, ...
    return len(keyword) > 2 and keyword[1] == "G" and not keyword.endswith(("GOR", "WGR"))


def _target_conversion(keyword: str, source: str, system: str,
                       gas_display: str = "Mscf") -> tuple[float, str, bool]:
    """Factor AFTER RSM header scale correction; unknown units are left unchanged.

    Surface-liquid: SM3 <-> STB; surface-gas: SM3 <-> Mscf/MMscf.
    Reservoir: RM3 <-> RB; pressure: BARSA <-> PSIA.
    Ratios: GOR SM3/SM3 <-> scf/STB; WGR SM3/SM3 <-> STB/MMscf.
    """
    src = source.strip().upper().replace(" ", "")
    src = src.replace("/DAYS", "/DAY").replace("/D", "/DAY") if src.endswith("/D") else src
    target_field = system == "Field"
    gas = _gas(keyword)
    gas_mult = (SCF_PER_M3 / (1_000_000 if gas_display == "MMscf" else 1000))
    gas_unit = "MMscf" if gas_display == "MMscf" else "Mscf"

    if src in ("BARSA", "BARA", "BAR", "PSIA", "PSI"):
        factor = PSI_PER_BAR if src.startswith("BAR") else 1.0
        return ((factor if target_field else factor / PSI_PER_BAR),
                "psia" if target_field else "bara", True)

    if src in ("SM3/SM3", "SCF/STB", "STB/MMSCF"):
        if keyword.endswith("GOR"):
            factor = SCF_PER_M3 / STB_PER_M3
            return (factor if src == "SM3/SM3" and target_field else
                    (1.0 if src == "SCF/STB" and target_field else
                     1.0 / factor if src == "SCF/STB" else 1.0),
                    "scf/STB" if target_field else "Sm3/Sm3", True)
        if keyword.endswith("WGR"):
            factor = STB_PER_M3 * 1_000_000 / SCF_PER_M3
            return (factor if src == "SM3/SM3" and target_field else
                    (1.0 if src == "STB/MMSCF" and target_field else
                     1.0 / factor if src == "STB/MMSCF" else 1.0),
                    "STB/MMscf" if target_field else "Sm3/Sm3", True)
        return (1., "fraction" if keyword.endswith("WCT") else "Sm3/Sm3", True)

    if src in ("SM3/DAY", "SM3", "STB/DAY", "STB", "MSCF/DAY", "MSCF",
               "MMSCF/DAY", "MMSCF", "SCF/DAY", "SCF"):
        rate = "/DAY" in src
        src_gas = src.startswith(("MSCF", "MMSCF", "SCF")) or (src.startswith("SM3") and gas)
        if src.startswith("SM3"):
            base_metric = 1.0
        elif src.startswith("STB"):
            base_metric = 1.0 / STB_PER_M3
        elif src.startswith("MMSCF"):
            base_metric = 1_000_000 / SCF_PER_M3
        elif src.startswith("MSCF"):
            base_metric = 1000 / SCF_PER_M3
        else:
            base_metric = 1.0 / SCF_PER_M3
        if src_gas:
            return (base_metric * gas_mult if target_field else base_metric,
                    gas_unit + ("/day" if rate else "") if target_field else
                    "Sm3" + ("/day" if rate else ""), True)
        return (base_metric * STB_PER_M3 if target_field else base_metric,
                "STB" + ("/day" if rate else "") if target_field else
                "Sm3" + ("/day" if rate else ""), True)

    if src in ("RM3/DAY", "RM3", "RB/DAY", "RB"):
        rate = "/DAY" in src
        base_metric = 1.0 if src.startswith("RM3") else 1.0 / STB_PER_M3
        return (base_metric * STB_PER_M3 if target_field else base_metric,
                ("RB" if target_field else "Rm3") + ("/day" if rate else ""), True)

    if src in ("SM3/D/B", "SM3/D/BAR", "STB/D/PSI"):
        base = 1.0 if src.startswith("SM3") else PSI_PER_BAR / STB_PER_M3
        return (base * STB_PER_M3 / PSI_PER_BAR if target_field else base,
                "STB/day/psi" if target_field else "Sm3/day/bar", True)

    if src in ("KJ/DAY", "MMBTU/DAY"):
        base = 1.0 if src.startswith("KJ") else 1_000_000 / BTU_PER_KJ
        return (base * BTU_PER_KJ / 1_000_000 if target_field else base,
                "MMBtu/day" if target_field else "kJ/day", True)

    if keyword.endswith("WCT") and src in ("%", "PERCENT"):
        return 0.01, "fraction", True
    if keyword.endswith("WCT") or keyword == "FOE":
        return 1.0, "fraction", True
    if keyword == "WSTAT":
        return 1.0, "status", True
    if src in ("YEARS", "DAYS"):
        return 1.0, src.lower(), True
    return 1.0, source or "dimensionless", bool(not source)


def value_and_unit(values: np.ndarray | pd.Series, meta: dict | pd.Series,
                   system: str, gas_display: str = "Mscf") -> tuple[np.ndarray, str]:
    factor, unit, _ = _target_conversion(str(meta["Keyword"]), str(meta["Source_Unit"]),
                                         system, gas_display)
    scale = float(meta["Scale_Factor"])
    return np.asarray(values, dtype="float64") * (scale * factor), unit


def parse_rsm(raw: bytes, filename: str = "case.RSM") -> dict[str, Any]:
    """Parse a tab-separated Eclipse RSM, reconstructing 10-column output pages.

    Returns: {'data': DataFrame(index=Time_Days, cols=internal IDs),
              'metadata': DataFrame, 'qc': dict}.
    """
    if not raw:
        raise ValueError("Empty RSM file")
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    start_positions = [i for i, s in enumerate(lines) if "SUMMARY OF RUN" in s.upper()]
    if not start_positions:
        raise ValueError("No 'SUMMARY OF RUN' headers found. Upload an ASCII Eclipse .RSM file, not binary .UNSMRY.")

    pages: list[pd.DataFrame] = []
    metadata = []
    issues: list[str] = []
    first_times = None
    all_ids = set()
    invalid_cells = 0
    mismatch_times = 0
    found_runs = set()

    for block_no, start in enumerate(start_positions):
        stop = start_positions[block_no + 1] if block_no + 1 < len(start_positions) else len(lines)
        m = re.search(r"SUMMARY OF RUN\s+(.+?)\s*$", lines[start], re.I)
        if m:
            found_runs.add(m.group(1).strip())
        segment = lines[start + 1:stop]
        if not segment:
            continue
        vrow = next((k for k, line in enumerate(segment) if _cells(line) and
                     _cells(line)[0].upper() in ("TIME", "DATE")), None)
        if vrow is None:
            issues.append(f"Block {block_no+1}: no TIME header")
            continue
        segment = segment[vrow:]
        labels = _cells(segment[0])
        ncols = len(labels)
        if ncols < 2:
            issues.append(f"Block {block_no+1}: no variables")
            continue
        header_rows = []
        number_start = None
        for idx, line in enumerate(segment[1:], start=1):
            cells = _cells(line)
            if cells and _is_number(cells[0]):
                number_start = idx
                break
            header_rows.append(cells)
        if number_start is None:
            issues.append(f"Block {block_no+1}: no numeric TIME rows")
            continue
        urow = header_rows[0] if header_rows else []
        xrows = header_rows[1:]
        ids = []
        for j, var in enumerate(labels[1:], start=1):
            keyword = var.upper().strip()
            if not keyword:
                ids.append(None)
                continue
            source_unit = urow[j].strip().upper() if len(urow) > j else ""
            extras = [r[j].strip() for r in xrows if len(r) > j and r[j].strip()]
            multipliers = [SCALE.search(x) for x in extras]
            scale_factor = float(np.prod([10.0 ** int(z.group(1)) for z in multipliers if z]))
            tags = [s for s in extras if not SCALE.search(s)]
            level = _scope(keyword)
            if level == "Well":
                entity = next((v for v in tags if not v.isdigit()), "UNKNOWN_WELL")
                qualifier = "" if entity != "UNKNOWN_WELL" else "?"
            elif level == "Group":
                entity = next((v for v in tags if not v.isdigit()), "UNKNOWN_GROUP")
                qualifier = ""
            else:
                entity = "FIELD"
                qualifier = "_".join(tags)
            variable = f"{keyword}__{qualifier}" if qualifier else keyword
            # Keep a distinct internal ID even if another section repeats a header.
            col_id = f"v{len(metadata):05d}"
            ids.append(col_id)
            metadata.append({
                "Column_ID": col_id,
                "Keyword": keyword,
                "Variable": variable,
                "Level": level,
                "Entity": entity,
                "Source_Unit": source_unit,
                "Scale_Factor": scale_factor,
                "Qualifier": qualifier,
                "Block": block_no + 1,
            })
        ts: list[float] = []
        rows: list[list[float]] = []
        for line in segment[number_start:]:
            v = _cells(line)
            if not v or not _is_number(v[0]):
                continue
            t = _num(v[0])
            if not np.isfinite(t):
                continue
            vals = []
            for j in range(1, ncols):
                s = v[j] if j < len(v) else ""
                value = _num(s)
                if np.isnan(value) and s.strip():
                    invalid_cells += 1
                vals.append(value)
            ts.append(t)
            rows.append(vals)
        if not ts:
            issues.append(f"Block {block_no+1}: empty time series")
            continue
        units_of_time = urow[0].strip().upper() if urow else "DAYS"
        times = np.array(ts, dtype="float64")
        if units_of_time in ("YEARS", "YEAR"):
            times *= 365.25
        if first_times is None:
            first_times = times
        elif len(first_times) != len(times) or not np.allclose(first_times, times, atol=1.e-4, rtol=0):
            mismatch_times += 1
        matrix = np.asarray(rows, dtype="float64")
        page = pd.DataFrame(matrix, index=pd.Index(times, name="Time_Days"),
                            columns=[x or f"unused_{block_no}_{j}" for j, x in enumerate(ids)])
        page = page.loc[:, [x for x in ids if x is not None]]
        if page.index.has_duplicates:
            issues.append(f"Block {block_no+1}: repeated TIME values (last retained)")
            page = page.loc[~page.index.duplicated(keep="last")]
        pages.append(page)

    if not pages:
        raise ValueError("No numerical summary pages could be decoded.")
    combined = pd.concat(pages, axis=1, join="outer", copy=False).sort_index()
    # Same column will never be reused: every page has its own internal ID.
    meta = pd.DataFrame.from_records(metadata)
    present = set(combined.columns)
    meta = meta[meta["Column_ID"].isin(present)].reset_index(drop=True)
    wells = meta.loc[meta.Level == "Well", "Entity"].nunique()
    result_qc = {
        "File": filename, "Bytes": len(raw), "Text_Lines": len(lines),
        "Blocks": len(pages), "Time_Steps": len(combined),
        "Time_Min_Days": float(combined.index.min()),
        "Time_Max_Days": float(combined.index.max()),
        "Columns": len(combined.columns),
        "Unique_Keywords": int(meta.Keyword.nunique()),
        "Wells": int(wells),
        "Time_Mismatched_Blocks": mismatch_times,
        "NonNumeric_Data_Cells": invalid_cells,
        "Scaled_Columns": int((meta.Scale_Factor != 1).sum()),
        "Run_Names": ", ".join(sorted(found_runs)),
        "Issues": issues,
    }
    return {"data": combined, "metadata": meta, "qc": result_qc}


def metadata_filter(parsed: dict, level: str | None = None,
                    keywords: Optional[Iterable[str]] = None,
                    entities: Optional[Iterable[str]] = None) -> pd.DataFrame:
    m = parsed["metadata"]
    mask = pd.Series(True, index=m.index)
    if level and level != "All":
        mask &= m["Level"].eq(level)
    if keywords is not None:
        mask &= m["Keyword"].isin(list(keywords))
    if entities is not None:
        mask &= m["Entity"].isin(list(entities))
    return m.loc[mask].copy()


def _date_index(time_days: np.ndarray, start_date: Optional[pd.Timestamp]) -> list[str]:
    if start_date is None:
        return [""] * len(time_days)
    return (pd.Timestamp(start_date) + pd.to_timedelta(time_days, unit="D")).strftime(
        "%Y-%m-%d %H:%M:%S").tolist()


def make_long_csv(cases: dict[str, dict], selection: dict[str, pd.DataFrame],
                  unit_system: str, gas_display: str, start_date=None) -> bytes:
    """Memory-conscious iterator builds Spotfire long CSV without melting >1m rows."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(["Case", "Date", "Time_Days", "Years", "Level", "Entity",
                     "Variable", "Keyword", "Unit", "Value"])
    for name, parsed in cases.items():
        df = parsed["data"]
        m = selection.get(name)
        if m is None or m.empty:
            continue
        td = df.index.to_numpy(dtype=float)
        dates = _date_index(td, start_date)
        year = td / 365.25
        # Column-wise writing: finite values only (do not manufacture zeroes).
        for r in m.itertuples(index=False):
            values, unit = value_and_unit(df[r.Column_ID].to_numpy(), r._asdict(),
                                          unit_system, gas_display)
            for d, t, y, v in zip(dates, td, year, values):
                if np.isfinite(v):
                    writer.writerow([name, d, float(t), float(y), r.Level, r.Entity,
                                     r.Variable, r.Keyword, unit, float(v)])
    return buf.getvalue().encode("utf-8-sig")


def make_wide_csv(cases: dict[str, dict], selection: dict[str, pd.DataFrame],
                  unit_system: str, gas_display: str, start_date=None) -> bytes:
    """One row per (Case, Time_Days, Level, Entity), each variable a column.

    Preserves per-column units in a companion Data Dictionary rather than
    writing mixed unit information into data rows. Converts all values first.
    """
    # Different entities/cases can have different variables; combine into one
    # frame. Empty fields are NaN, never interpreted as zero.
    pieces = []
    for case, parsed in cases.items():
        frame = parsed["data"]
        meta = selection.get(case)
        if meta is None or meta.empty:
            continue
        ts = frame.index.to_numpy(dtype=float)
        days_dates = _date_index(ts, start_date)
        for (level, entity), sub in meta.groupby(["Level", "Entity"], sort=False):
            part = pd.DataFrame({
                "Case": case,
                "Date": days_dates,
                "Time_Days": ts,
                "Years": ts / 365.25,
                "Level": level,
                "Entity": entity,
            })
            for r in sub.itertuples(index=False):
                arr, _ = value_and_unit(frame[r.Column_ID].to_numpy(), r._asdict(),
                                        unit_system, gas_display)
                name = r.Variable
                if name in part.columns:
                    name += "__duplicate"
                part[name] = arr
            pieces.append(part)
    if not pieces:
        return b""
    wide = pd.concat(pieces, ignore_index=True, sort=False)
    return wide.to_csv(index=False).encode("utf-8-sig")


def make_dictionary_csv(cases: dict[str, dict], selection: dict[str, pd.DataFrame],
                        unit_system: str, gas_display: str) -> bytes:
    data = []
    for name, parsed in cases.items():
        sub = selection.get(name)
        if sub is None:
            continue
        for r in sub.itertuples(index=False):
            _, target_unit, supported = _target_conversion(r.Keyword, r.Source_Unit,
                                                            unit_system, gas_display)
            data.append({"Case": name, "Level": r.Level, "Entity": r.Entity,
                         "Variable": r.Variable, "Keyword": r.Keyword,
                         "Original_Unit": r.Source_Unit,
                         "Original_RSM_Scale": r.Scale_Factor,
                         "Export_Unit": target_unit,
                         "Conversion_Supported": supported})
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8-sig")
