"""Eclipse RSM Studio — Single-file GitHub / Streamlit Cloud app.
Upload this script and requirements.txt to your repo, choose streamlit_app.py as main.
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


# Descriptive names are kept separate from ECLIPSE keywords so exports remain stable.
# Vectors with less certain semantics retain neutral names rather than misleading labels.
KEYWORD_NAMES = {
    "FPR": "Field Average Reservoir Pressure", "WBHP": "Well Bottomhole Pressure",
    "WTHP": "Well Tubing Head Pressure", "WBP9": "Well Block Pressure (WBP9)",
    "WPI": "Well Productivity Index", "WSTAT": "Well Status",
    "FOE": "Field Oil Recovery Efficiency", "FRS": "Field Ratio / RSM Summary",
    "FRV": "Field Reservoir Volume Ratio", "TIMESTEP": "Simulation Time Step",
    "YEARS": "Simulation Years", "AAQP": "Aquifer Pressure",
    "AAQR": "Aquifer Flow Rate", "AAQT": "Cumulative Aquifer Flow",
    "AAQPD": "Aquifer Diagnostic (AAQPD)", "AAQTD": "Aquifer Diagnostic (AAQTD)",
    "WEPR": "Well Energy Production Rate",
}
_FLUID_NAMES = {"O": "Oil", "W": "Water", "G": "Gas", "V": "Reservoir Volume"}
_ACTIVITY_NAMES = {
    "PR": "Production Rate", "PT": "Cumulative Production", "IR": "Injection Rate",
    "IT": "Cumulative Injection", "IP": "In Place", "CT": "Cut",
    "OR": "Oil-Gas Ratio", "GR": "Water-Gas Ratio",
}


def keyword_description(keyword: str) -> str:
    """Human-readable descriptor without changing the RSM's original keyword."""
    k = str(keyword).upper().strip()
    if k in KEYWORD_NAMES:
        return KEYWORD_NAMES[k]
    if len(k) >= 4 and k[0] in 'FWG' and k[1] in _FLUID_NAMES:
        level = {'F': 'Field', 'W': 'Well', 'G': 'Group'}[k[0]]
        fluid = _FLUID_NAMES[k[1]]
        tail = k[2:]
        if tail in _ACTIVITY_NAMES:
            activity = _ACTIVITY_NAMES[tail]
            if tail == 'OR':
                return f"{level} Gas-Oil Ratio"  # xGOR = GOR (not oil/gas)
            if tail == 'GR':
                return f"{level} Water-Gas Ratio"
            return f"{level} {fluid} {activity}"
        if tail == 'IPG':
            return f"{level} {fluid} In Place (Gas Phase)"
        if tail == 'IPL':
            return f"{level} {fluid} In Place (Liquid Phase)"
    return f"ECLIPSE Summary Variable ({k})"


def keyword_label(keyword: str) -> str:
    return f"{keyword} — {keyword_description(keyword)}"


def fluid_family(keyword: str) -> str:
    k = str(keyword).upper()
    if k in ('FPR', 'WBHP', 'WTHP', 'WBP9', 'AAQP'):
        return 'Pressure'
    if k.endswith(('WCT', 'WGR')) or (len(k) > 1 and k[1] == 'W'):
        return 'Water'
    if k.endswith('GOR') or (len(k) > 1 and k[1] == 'G'):
        return 'Gas'
    if len(k) > 1 and k[1] == 'O':
        return 'Oil'
    return 'Other'


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
                     "Variable", "Keyword", "Column_Name", "Unit", "Value"])
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
                                     r.Variable, r.Keyword, keyword_description(r.Keyword), unit, float(v)])
    return buf.getvalue().encode("utf-8-sig")


def make_wide_csv(cases: dict[str, dict], selection: dict[str, pd.DataFrame],
                  unit_system: str, gas_display: str, start_date=None,
                  descriptive_headers: bool = False) -> bytes:
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
                name = (f"{r.Variable} — {keyword_description(r.Keyword)}"
                        if descriptive_headers else r.Variable)
                base_name = name
                duplicate = 2
                while name in part.columns:
                    name = f"{base_name}__duplicate_{duplicate}"
                    duplicate += 1
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
                         "Column_Name": keyword_description(r.Keyword),
                         "Original_Unit": r.Source_Unit,
                         "Original_RSM_Scale": r.Scale_Factor,
                         "Export_Unit": target_unit,
                         "Conversion_Supported": supported})
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8-sig")

# ---- Streamlit user interface ----
"""Modern single-file Streamlit app. No local Python imports are required."""
from datetime import date
from pathlib import Path
import html
import math
import zipfile

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title='Eclipse RSM Studio Pro', page_icon='📊', layout='wide')

DEFAULT_PALETTE = {
    'Oil': '#16A34A', 'Gas': '#DC2626', 'Water': '#2563EB',
    'Pressure': '#D97706', 'Other': '#7C3AED',
}
CASE_COLORS = px.colors.qualitative.Safe
ENTITY_COLORS = px.colors.qualitative.Bold
LINE_DASHES = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']


def time_axis(days, mode, start_date):
    if mode == 'Years':
        return np.asarray(days, dtype=float) / 365.25
    if mode == 'Date' and start_date is not None:
        return pd.Timestamp(start_date) + pd.to_timedelta(days, unit='D')
    return np.asarray(days, dtype=float)


def axis_label(mode):
    return {'Time (days)': 'Simulation time (days)', 'Years': 'Simulation time (years)',
            'Date': 'Calendar date'}.get(mode, 'Simulation time')


def trace_data(parsed, keyword, level, entity, unit_system, gas_display):
    meta = metadata_filter(parsed, level, [keyword], [entity])
    if meta.empty:
        return None
    r = meta.iloc[0]
    values, unit = value_and_unit(parsed['data'][r.Column_ID].to_numpy(), r, unit_system, gas_display)
    return parsed['data'].index.to_numpy(dtype=float), values, unit


def style_markup(text, style):
    text = html.escape(str(text))
    if 'Bold' in style:
        text = '<b>' + text + '</b>'
    if 'Italic' in style:
        text = '<i>' + text + '</i>'
    return text


def rgba_from_hex(value, opacity):
    """Apply line/area opacity to Plotly hex, rgb() and rgba() colors."""
    value = str(value).strip()
    h = value.lstrip('#')
    if re.fullmatch('[0-9a-fA-F]{6}', h):
        parts = (int(h[i:i+2],16) for i in (0,2,4))
    elif value.startswith(('rgb(', 'rgba(')):
        nums = re.findall(r'[0-9.]+', value)
        if len(nums) < 3:
            return value
        parts = tuple(int(float(v)) for v in nums[:3])
    else:
        return value
    red,green,blue = parts
    return f'rgba({red},{green},{blue},{float(opacity):.3f})'


def collect_plot_records(cases, level, keywords, entities, system, gas_display,
                         watercut_percent=True, maximum=250):
    """Collection for graphing only. Full CSV export is never truncated."""
    records = []
    for case, parsed in cases.items():
        for key in keywords:
            for entity in entities:
                if len(records) >= maximum:
                    return records, True
                series = trace_data(parsed, key, level, entity, system, gas_display)
                if series is None:
                    continue
                days, values, unit = series
                if key.endswith('WCT') and watercut_percent:
                    values = values * 100
                    unit = '%'
                records.append(dict(case=case, entity=entity, key=key,
                                    days=days, values=values, unit=unit))
    return records, False


def plot_color(rec, color_by, palette, cases, entities, keywords):
    if color_by == 'Case':
        return CASE_COLORS[list(cases).index(rec['case']) % len(CASE_COLORS)]
    if color_by == 'Well / Entity':
        return ENTITY_COLORS[list(entities).index(rec['entity']) % len(ENTITY_COLORS)]
    if color_by == 'Keyword':
        return CASE_COLORS[list(keywords).index(rec['key']) % len(CASE_COLORS)]
    return palette[fluid_family(rec['key'])]


def chart_factory(records, mode, start_date, level, cases, entities, keywords,
                  palette, settings, trellis_by=None, nrows=1, ncols=1, page=1,
                  render_backend='webgl'):
    """Overlay or paged Trellis with at most two axes per panel.

    SVG is used for Overview charts. Inactive Streamlit tabs remain mounted,
    and many Scattergl figures can exhaust browser WebGL contexts after
    viewing Trellis charts, leaving visible axes but invisible data curves.
    """
    if render_backend not in ('webgl', 'svg'):
        raise ValueError("render_backend must be 'webgl' or 'svg'")
    scatter_type = go.Scatter if render_backend == 'svg' else go.Scattergl
    facets = ['ALL']
    if trellis_by == 'Case':
        facets = list(dict.fromkeys(r['case'] for r in records))
    elif trellis_by == 'Well / Entity':
        facets = list(dict.fromkeys(r['entity'] for r in records))
    elif trellis_by == 'Keyword':
        facets = list(dict.fromkeys(r['key'] for r in records))
    is_trellis = trellis_by is not None
    if not is_trellis:
        nrows, ncols = 1, 1
    cells = nrows * ncols
    active_facets = facets[(page-1)*cells:page*cells]
    if not active_facets:
        active_facets = ['ALL']
    rows_used = math.ceil(len(active_facets)/ncols)
    fig = make_subplots(
        rows=rows_used, cols=ncols,
        specs=[[{'secondary_y': True} for _ in range(ncols)] for _ in range(rows_used)],
        subplot_titles=([style_markup(keyword_label(s) if trellis_by == 'Keyword' else s,
                                     settings['Font style']) for s in active_facets]
                        if is_trellis else None),
        shared_xaxes='all' if is_trellis else False,
        vertical_spacing=min(0.16, max(0.045, 0.30/rows_used)),
        horizontal_spacing=min(0.11, max(0.045, 0.18/ncols)),
    )
    units_by_panel = {}
    skipped_units = set()
    seen_legends = set()
    num_traces = 0
    for idx, facet in enumerate(active_facets):
        row = idx // ncols + 1
        col = idx % ncols + 1
        panel = [r for r in records if (
            facet == 'ALL' or
            (trellis_by == 'Case' and r['case'] == facet) or
            (trellis_by == 'Well / Entity' and r['entity'] == facet) or
            (trellis_by == 'Keyword' and r['key'] == facet))]
        local_units = list(dict.fromkeys(r['unit'] for r in panel))
        units_by_panel[(row, col)] = local_units[:2]
        if len(local_units) > 2:
            skipped_units.update(local_units[2:])
        for r in panel:
            if r['unit'] not in local_units[:2]:
                continue
            second = len(local_units) > 1 and r['unit'] == local_units[1]
            name = f"{r['case']} · {r['entity']} · {r['key']}"
            if level == 'Field':
                name = f"{r['case']} · {r['key']}"
            label = keyword_label(r['key'])
            key = (r['case'], r['entity'], r['key'])
            color = plot_color(r, settings['Color by'], palette, cases, entities, keywords)
            dash_id = list(cases).index(r['case']) + list(entities).index(r['entity'])
            linestyle = LINE_DASHES[dash_id % len(LINE_DASHES)] if settings['Distinguish lines'] else 'solid'
            y = r['values']
            x = time_axis(r['days'], mode, start_date)
            if len(y) > settings['Max chart points']:
                keep = np.unique(np.linspace(0, len(y)-1, settings['Max chart points'], dtype=int))
                x, y = x[keep], y[keep]
            trace = scatter_type(
                x=x, y=y, mode='lines', name=name,
                legendgroup=name, showlegend=name not in seen_legends,
                line=dict(color=rgba_from_hex(color, settings['Line opacity']),
                          width=settings['Line width'], dash=linestyle),
                connectgaps=False,
                fill='tozeroy' if settings['Area fill'] and not second else None,
                fillcolor=rgba_from_hex(color, settings['Area opacity']),
                hovertemplate=(f"<b>{html.escape(r['case'])}</b> · {html.escape(r['entity'])}"
                               f"<br>{html.escape(label)}"
                               f"<br>{axis_label(mode)}: %{{x}}"
                               f"<br>Value: %{{y:,.4g}} {html.escape(r['unit'])}<extra></extra>"),
            )
            seen_legends.add(name)
            fig.add_trace(trace, row=row, col=col, secondary_y=second)
            num_traces += 1
        if local_units:
            fig.update_yaxes(title_text=style_markup(local_units[0], settings['Font style'])
                             if col == 1 else None,
                             row=row, col=col, secondary_y=False)
            if len(local_units) > 1:
                fig.update_yaxes(title_text=style_markup(local_units[1], settings['Font style'])
                                 if col == ncols else None,
                                 row=row, col=col, secondary_y=True)
        if row == rows_used:
            fig.update_xaxes(title_text=style_markup(axis_label(mode), settings['Font style']),
                             row=row, col=col)
    is_dark = settings['Theme'] == 'Dark'
    base_text = '#E2E8F0' if is_dark and not settings['Transparent background'] else '#334155'
    grid = (f"rgba(148,163,184,{settings['Grid opacity']:.2f})"
            if settings['Show grid'] else 'rgba(0,0,0,0)')
    paper = 'rgba(0,0,0,0)' if settings['Transparent background'] else ('#0F172A' if is_dark else '#FFFFFF')
    plot = 'rgba(0,0,0,0)' if settings['Transparent background'] else ('#111827' if is_dark else '#FFFFFF')
    font_family = 'Arial Black, Arial, sans-serif' if 'Bold' in settings['Font style'] else ('Georgia, serif' if 'Italic' in settings['Font style'] else 'Arial, sans-serif')
    fig.update_layout(
        template='plotly_dark' if is_dark else 'plotly_white',
        title=dict(text=style_markup(settings['Chart title'], settings['Font style']),
                   x=0.015, xanchor='left', font=dict(size=settings['Font size']+5)),
        font=dict(family=font_family, size=settings['Font size'], color=base_text),
        paper_bgcolor=paper, plot_bgcolor=plot,
        height=max(360, settings['Panel height']*rows_used+150),
        margin=dict(t=95 if is_trellis else 85, b=115, l=65, r=65),
        legend=dict(orientation='h', yanchor='top', y=-0.16, xanchor='left', x=0,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=max(9,settings['Font size']-1))),
        hovermode='closest' if is_trellis else 'x unified',
    )
    fig.update_xaxes(showgrid=settings['Show grid'], gridcolor=grid,
                     zeroline=False, showline=True, linecolor=grid,
                     tickfont=dict(size=max(settings['Font size']-1, 9)))
    fig.update_yaxes(showgrid=settings['Show grid'], gridcolor=grid,
                     zeroline=False, showline=False,
                     tickfont=dict(size=max(settings['Font size']-1, 9)))
    if fig.layout.annotations:
        fig.update_annotations(font=dict(size=max(10, settings['Font size']+1), color=base_text))
    return fig, num_traces, skipped_units


def summarize_last(cases, level, keywords, entities, system, gas_display):
    recs = []
    for case, parsed in cases.items():
        for key in keywords:
            for entity in entities:
                curve = trace_data(parsed, key, level, entity, system, gas_display)
                if curve is None:
                    continue
                days, y, unit = curve
                idx = np.flatnonzero(np.isfinite(y))
                if len(idx):
                    i = idx[-1]
                    recs.append(dict(Case=case, Level=level, Entity=entity,
                                     Keyword=key, Column_Name=keyword_description(key),
                                     Time_Days=float(days[i]), Value=float(y[i]), Unit=unit))
    return pd.DataFrame(recs)


def preview_records(cases, level, keywords, entities, system, gas_display,
                    start_date, limit=3000):
    chunks = []
    remaining = limit
    for case, parsed in cases.items():
        m = metadata_filter(parsed, level, keywords, entities)
        td = parsed['data'].index.to_numpy(dtype=float)
        for r in m.itertuples(index=False):
            if remaining <= 0:
                break
            v, unit = value_and_unit(parsed['data'][r.Column_ID].to_numpy(),
                                     r._asdict(), system, gas_display)
            n = min(remaining, len(td))
            part = pd.DataFrame({'Case': case, 'Time_Days': td[:n],
                                 'Years': td[:n]/365.25, 'Level': r.Level,
                                 'Entity': r.Entity, 'Keyword': r.Keyword,
                                 'Column_Name': keyword_description(r.Keyword),
                                 'Variable': r.Variable, 'Unit': unit, 'Value': v[:n]})
            if start_date is not None:
                part.insert(1, 'Date', pd.Timestamp(start_date) + pd.to_timedelta(td[:n], unit='D'))
            chunks.append(part)
            remaining -= n
        if remaining <= 0:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=8)
def cached_parse(raw: bytes, filename: str):
    return parse_rsm(raw, filename)


st.markdown('''
<style>
.block-container {padding-top: 1.2rem; max-width: 1750px;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stMetricValue"] {font-size:1.8rem; font-weight:700;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
.rsm-subtitle {color:#64748b;font-size:1rem;margin-top:-.5rem;margin-bottom:1.2rem;}
.rsm-note {color:#64748b;font-size:.83rem;}
</style>
''', unsafe_allow_html=True)

st.title('Eclipse RSM Studio Pro')
st.markdown('<div class="rsm-subtitle">Multi-case Eclipse summary viewer · Spotfire-style Trellis · Engineering units · Spotfire CSV</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('1 · Eclipse cases')
    uploads = st.file_uploader('Upload one or more .RSM files', type=['rsm','txt'],
                               accept_multiple_files=True,
                               help='Each uploaded RSM becomes a named simulation case.')
    st.header('2 · Engineering units')
    unit_system = st.radio('Unit system', ['Field','Metric'], horizontal=True)
    gas_display = st.selectbox('Field gas units', ['Mscf','MMscf']) if unit_system == 'Field' else 'Mscf'
    st.header('3 · Simulation timeline')
    has_start = st.checkbox('Set calendar start date', value=False)
    start_date = st.date_input('Simulation day 0', value=date.today()) if has_start else None
    x_mode = st.radio('X-axis', ['Time (days)','Years'] + (['Date'] if has_start else []), horizontal=True)
    st.caption('Only enter a date if you know the ECLIPSE simulation start date.')

if not uploads:
    st.info('Upload Eclipse ASCII .RSM summary file(s) using the sidebar to begin.')
    st.markdown('**Features:** Field / Well / Group profiles; multiple simultaneous vectors; '
                'editable Oil / Gas / Water colors; Spotfire Trellis; ranking; merged Long / Wide / Dictionary CSV.')
    st.stop()

cases = {}
original_names = {}
failed = []
with st.sidebar:
    st.header('4 · Case labels')
    for i, upload in enumerate(uploads):
        default_label = Path(upload.name).stem
        label = st.text_input(f'Case {i+1}: {upload.name}', value=default_label,
                              key=f'casename_{i}_{upload.name}').strip() or default_label
        if label in cases:
            label = f'{label} ({i+1})'
        try:
            with st.spinner(f'Parsing {upload.name}...'):
                cases[label] = cached_parse(upload.getvalue(), upload.name)
            original_names[label] = upload.name
        except Exception as exc:
            failed.append(f'{upload.name}: {exc}')

for error in failed:
    st.error(error)
if not cases:
    st.stop()

meta_all = pd.concat([parsed['metadata'] for parsed in cases.values()], ignore_index=True)
levels = [x for x in ['Field','Well','Group'] if x in set(meta_all.Level)]
all_wells = sorted(meta_all.loc[meta_all.Level.eq('Well'), 'Entity'].unique().tolist())

with st.sidebar:
    with st.expander('5 · Plot appearance / Color Studio', expanded=False):
        palette = {}
        st.caption('Default engineering colors; choose any color below.')
        palette['Oil'] = st.color_picker('Oil · Green', DEFAULT_PALETTE['Oil'])
        palette['Water'] = st.color_picker('Water · Blue', DEFAULT_PALETTE['Water'])
        palette['Gas'] = st.color_picker('Gas · Red', DEFAULT_PALETTE['Gas'])
        palette['Pressure'] = st.color_picker('Pressure · Amber', DEFAULT_PALETTE['Pressure'])
        palette['Other'] = st.color_picker('Other', DEFAULT_PALETTE['Other'])
        theme = st.selectbox('Plot theme', ['Light','Dark'])
        transparent = st.checkbox('Transparent plot background', value=True)
        font_style = st.selectbox('Font style', ['Normal','Bold','Italic','Bold Italic'], index=1)
        font_size = st.slider('Font size (pt)', min_value=10, max_value=22, value=12, step=1)
        line_width = st.slider('Line width', min_value=1.0, max_value=5.0, value=2.5, step=0.5)
        line_opacity = st.slider('Line opacity', min_value=0.30, max_value=1.0,
                                 value=0.95, step=0.05)
        area_fill = st.checkbox('Soft area fill under lines', value=False)
        area_opacity = st.slider('Area fill opacity', min_value=0.05, max_value=0.50,
                                 value=0.12, step=0.05) if area_fill else 0.12
        show_grid = st.checkbox('Show subtle gridlines', value=True)
        grid_opacity = st.slider('Grid opacity', min_value=0.0, max_value=0.50,
                                 value=0.16, step=0.05) if show_grid else 0.0
        distinguish_lines = st.checkbox('Different line styles for cases/wells', value=True)
        panel_height = st.slider('Height per Trellis row (px)', min_value=220,
                                 max_value=650, value=340, step=20)
        chart_point_max = st.number_input('Max plotted points per curve', min_value=200,
                                          max_value=30000, value=5000, step=200)

settings = {'Theme': theme, 'Transparent background': transparent, 'Font style': font_style,
            'Font size': font_size, 'Line width': line_width, 'Line opacity': line_opacity,
            'Area fill': area_fill, 'Area opacity': area_opacity,
            'Show grid': show_grid, 'Grid opacity': grid_opacity,
            'Distinguish lines': distinguish_lines, 'Panel height': panel_height,
            'Max chart points': int(chart_point_max), 'Chart title': 'Production profiles',
            'Color by': 'Fluid / Metric'}

s1,s2,s3,s4 = st.columns(4)
s1.metric('Loaded cases', str(len(cases)))
s2.metric('Unique wells', f'{len(all_wells):,}')
s3.metric('ECLIPSE keywords', str(meta_all.Keyword.nunique()))
s4.metric('Longest simulation', f"{max(p['qc']['Time_Max_Days'] for p in cases.values()):,.0f} days")

if any(p['qc']['Time_Mismatched_Blocks'] for p in cases.values()):
    st.warning('One or more RSM blocks have different TIME values; an outer join on simulation time was used.')

tab_overview,tab_plots,tab_rank,tab_data,tab_export,tab_qc = st.tabs([
    'Overview','Interactive plots & Trellis','Well ranking','Data explorer','Spotfire export','QC & dictionary'])

with tab_overview:
    st.subheader('Field summary — last available timestep')
    main_fields = ['FOPR','FWPR','FGPR','FOPT','FWPT','FGPT','FWCT','FPR']
    summary = []
    for case, parsed in cases.items():
        rec = {'Case': case, 'Simulation end (days)': parsed['qc']['Time_Max_Days']}
        for kw in main_fields:
            series = trace_data(parsed, kw, 'Field', 'FIELD', unit_system, gas_display)
            if series is None:
                continue
            _,v,unit = series
            finite = v[np.isfinite(v)]
            rec[f'{kw} · {keyword_description(kw)} [{unit}]'] = float(finite[-1]) if len(finite) else np.nan
        summary.append(rec)
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.caption('End-of-run values (not sums over time). ECLIPSE printed ×10ⁿ scale factors are applied.')
    st.subheader('Field production dashboard')
    dash_keys = [('FOPR','Oil rate'),('FWPR','Water rate'),('FGPR','Gas rate'),
                 ('FOPT','Cumulative oil'),('FWCT','Water cut'),('FPR','Average reservoir pressure')]
    field_keys = set(meta_all.loc[meta_all.Level.eq('Field'),'Keyword'])
    for n in range(0,len(dash_keys),2):
        left,right = st.columns(2)
        for col,(kw,caption) in zip((left,right),dash_keys[n:n+2]):
            if kw not in field_keys:
                continue
            with col:
                # A native Streamlit heading stays visible when hidden tabs
                # resize; Plotly's in-figure title could become clipped.
                st.markdown(f'#### {caption} · {kw}')
                records,_ = collect_plot_records(cases,'Field',[kw],['FIELD'],unit_system,gas_display)
                opts = settings.copy()
                opts['Chart title'] = f'{caption} · {kw}'
                fig,count,_ = chart_factory(
                    records,x_mode,start_date,'Field',list(cases),['FIELD'],[kw],
                    palette,opts,render_backend='svg'
                )
                # SVG curves survive WebGL context exhaustion after visiting
                # heavy Plotly Trellis pages. Place legend clear of x-axis.
                fig.update_layout(
                    title=None,
                    height=440,
                    autosize=True,
                    margin=dict(t=20,b=155,l=65,r=40),
                    legend=dict(
                        orientation='h',
                        x=0.5,xanchor='center',y=-0.33,yanchor='top',
                        bgcolor='rgba(0,0,0,0)',
                        font=dict(size=max(9,settings['Font size']-1)),
                    ),
                    uirevision=False,
                )
                fig.update_xaxes(automargin=True)
                fig.update_yaxes(automargin=True)
                st.plotly_chart(
                    fig,use_container_width=True,key=f'overview_{kw}',
                    config={'displaylogo':False,'responsive':True},
                )

with tab_plots:
    st.subheader('Interactive engineering plots')
    c1,c2,c3 = st.columns([1,2,1.5])
    with c1:
        chart_level = st.selectbox('Analysis level', levels, key='plot_level')
    available_kw = sorted(meta_all.loc[meta_all.Level.eq(chart_level),'Keyword'].unique().tolist())
    suggested = {'Field':['FOPR','FWPR','FGPR'], 'Well':['WOPR','WWPR','WGPR'],
                 'Group':['GOPR','GWPR','GGPR']}.get(chart_level,[])
    with c2:
        chart_kw = st.multiselect('ECLIPSE vectors · full column names', available_kw,
                                  default=[k for k in suggested if k in available_kw],
                                  format_func=keyword_label,
                                  key=f'plot_keys_{chart_level}')
    with c3:
        case_names = st.multiselect('Simulation cases', list(cases), default=list(cases),
                                    key='plot_cases')
    if chart_level == 'Field':
        available_entities = ['FIELD']
        entities = ['FIELD']
    else:
        available_entities = sorted(meta_all.loc[(meta_all.Level == chart_level) &
                              (meta_all.Keyword.isin(chart_kw)), 'Entity'].unique().tolist())
        entities = st.multiselect(f'{chart_level}s / entities', available_entities,
                                  default=available_entities[:5], key=f'plot_entities_{chart_level}')
    c4,c5,c6 = st.columns([1.2,1.4,2])
    with c4:
        plot_layout = st.radio('Plot layout', ['Overlay','Trellis'], horizontal=True)
    with c5:
        color_by = st.selectbox('Color by', ['Fluid / Metric','Case','Well / Entity','Keyword'])
    with c6:
        custom_title = st.text_input('Chart title', value='RSM Production Profiles')
    settings['Chart title'] = custom_title
    settings['Color by'] = color_by
    percent_wct = st.checkbox('Display water cut (%) on plots — CSV exports retain fractions', value=True)
    trellis_by = None
    ncols,nrows,page = 1,1,1
    subset = {c: cases[c] for c in case_names}
    if plot_layout == 'Trellis':
        t1,t2,t3 = st.columns(3)
        with t1:
            trellis_options = (['Well / Entity','Case','Keyword'] if chart_level != 'Field'
                               else ['Case','Keyword'])
            trellis_by = st.selectbox('Trellis by', trellis_options, key='trellis_dimension')
        with t2:
            ncols = st.number_input('Horizontal · plots per row',min_value=1,max_value=6,value=3,step=1)
        with t3:
            nrows = st.number_input('Vertical · plot rows',min_value=1,max_value=6,value=2,step=1)
    if chart_kw and case_names and entities:
        records,truncated = collect_plot_records(subset, chart_level, chart_kw, entities,
                                                 unit_system,gas_display,percent_wct)
        if records:
            if truncated:
                st.warning('Plot limited to the first 250 curves for responsive rendering. CSV exports are not limited.')
            facet_values = ['ALL']
            if trellis_by == 'Case':
                facet_values = list(dict.fromkeys(r['case'] for r in records))
            elif trellis_by == 'Well / Entity':
                facet_values = list(dict.fromkeys(r['entity'] for r in records))
            elif trellis_by == 'Keyword':
                facet_values = list(dict.fromkeys(r['key'] for r in records))
            total_pages = max(1,math.ceil(len(facet_values)/int(nrows*ncols)))
            if plot_layout == 'Trellis':
                st.caption(f'{len(facet_values)} Trellis panels · {ncols} horizontal × {nrows} vertical · {total_pages} page(s)')
                if total_pages > 1:
                    page = st.number_input('Trellis page',min_value=1,max_value=total_pages,
                                           value=1,step=1,key=f'trellis_page_{trellis_by}_{ncols}_{nrows}')
            fig,n,skipped = chart_factory(records,x_mode,start_date,chart_level,case_names,
                                          entities,chart_kw,palette,settings,trellis_by,
                                          int(nrows),int(ncols),int(page))
            if skipped:
                st.warning('A panel supports at most two different units on dual Y-axes. '
                           f'Unplotted units: {", ".join(sorted(skipped))}. '
                           'Use Trellis by Keyword to show all vectors separately.')
            st.plotly_chart(fig,use_container_width=True,key='main_rsm_plot',
                            config={'displaylogo':False,'scrollZoom':True,
                                    'toImageButtonOptions':{'format':'png','filename':'Eclipse_RSM_Plot','scale':2}})
            st.download_button('Download interactive Plotly chart (HTML)',
                               data=fig.to_html(full_html=True,include_plotlyjs=True).encode('utf-8'),
                               file_name='Eclipse_RSM_Chart.html',mime='text/html',key='download_html')
            st.caption(f'{n} plotted lines · Lines color-coded by {color_by.lower()} · Hover to inspect values. '
                        'Use the Plotly camera icon to export PNG.')
            st.markdown('**Last recorded values — selected vectors**')
            last = summarize_last(subset,chart_level,chart_kw,entities,unit_system,gas_display)
            if not last.empty:
                st.dataframe(last,use_container_width=True,hide_index=True,height=340)
        else:
            st.info('No matching summary vectors exist for the current case / well selection.')
    else:
        st.info('Choose at least one case, keyword, and entity to plot.')

with tab_rank:
    st.subheader('Well ranking · choose a simulation day')
    well_kw = sorted(meta_all.loc[meta_all.Level.eq('Well'),'Keyword'].unique().tolist())
    if well_kw:
        col1,col2,col3 = st.columns(3)
        with col1:
            ranking_kw = st.selectbox('Ranking keyword',well_kw,
                                      index=well_kw.index('WOPR') if 'WOPR' in well_kw else 0,
                                      format_func=keyword_label)
        lo = float(min(p['qc']['Time_Min_Days'] for p in cases.values()))
        hi = float(max(p['qc']['Time_Max_Days'] for p in cases.values()))
        with col2:
            # Streamlit sliders reject a float range with an integer step.
            # number_input uses homogeneous native float parameters and supports decimals.
            ranking_day = st.number_input('Simulation day', min_value=lo,max_value=hi,
                                          value=hi,step=1.0,format='%.2f')
        with col3:
            top_n = st.number_input('Top wells per case',min_value=1,
                                    max_value=max(1,len(all_wells)),value=min(20,max(1,len(all_wells))),step=1)
        data=[]
        for case,parsed in cases.items():
            td=parsed['data'].index.to_numpy(dtype=float)
            ix=int(np.abs(td-ranking_day).argmin())
            for r in metadata_filter(parsed,'Well',[ranking_kw]).itertuples(index=False):
                v,unit=value_and_unit(parsed['data'][r.Column_ID].to_numpy(),r._asdict(),
                                       unit_system,gas_display)
                if np.isfinite(v[ix]):
                    data.append({'Case':case,'Well':r.Entity,'Time_Days':float(td[ix]),
                                 'Keyword':ranking_kw,'Column_Name':keyword_description(ranking_kw),
                                 'Value':float(v[ix]),'Unit':unit})
        if data:
            ranking=pd.DataFrame(data)
            ranking=(ranking.sort_values('Value',ascending=False)
                    .groupby('Case',group_keys=False).head(int(top_n)))
            fig=px.bar(ranking,x='Well',y='Value',color='Case',barmode='group',
                       title=keyword_label(ranking_kw),template='plotly_white',
                       labels={'Value':ranking_kw+' ('+str(ranking.iloc[0]['Unit'])+')'},
                       hover_data=['Time_Days'])
            fig.update_layout(height=490,xaxis_tickangle=-45,
                              font=dict(size=font_size,family='Arial Black' if 'Bold' in font_style else 'Arial'),
                              paper_bgcolor='rgba(0,0,0,0)' if transparent else '#FFFFFF',
                              plot_bgcolor='rgba(0,0,0,0)' if transparent else '#FFFFFF')
            st.plotly_chart(fig,use_container_width=True,key='ranking_bar',config={'displaylogo':False})
            st.dataframe(ranking,use_container_width=True,hide_index=True)
            st.download_button('Download well ranking CSV',ranking.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f'Well_Ranking_{ranking_kw}.csv',mime='text/csv')
        else:
            st.info('No well data available for this simulation day.')
    else:
        st.info('No Well-level summary vectors were detected.')

with tab_data:
    st.subheader('Tabular data explorer')
    a,b=st.columns([1,2])
    with a:
        explorer_level=st.selectbox('Analysis level',levels,key='explorer_level')
    av=sorted(meta_all.loc[meta_all.Level.eq(explorer_level),'Keyword'].unique().tolist())
    pref={'Field':['FOPR','FWPR','FGPR','FOPT'],
          'Well':['WOPR','WWPR','WGPR','WOPT'],
          'Group':['GOPR','GWPR','GGPR']}.get(explorer_level,[])
    with b:
        explorer_kw=st.multiselect('Keyword · full name',av,
                                   default=[k for k in pref if k in av],
                                   format_func=keyword_label,key=f'expl_kw_{explorer_level}')
    if explorer_level=='Field':
        explorer_ent=['FIELD']
    else:
        ent_options=sorted(meta_all.loc[meta_all.Level.eq(explorer_level),'Entity'].unique().tolist())
        explorer_ent=st.multiselect('Entities',ent_options,default=ent_options[:3],
                                    key=f'expl_ent_{explorer_level}')
    st.caption('Preview capped at 3,000 rows. All raw timesteps remain available in CSV export.')
    if explorer_kw and explorer_ent:
        tab=preview_records(cases,explorer_level,explorer_kw,explorer_ent,
                            unit_system,gas_display,start_date)
        if not tab.empty:
            st.dataframe(tab,use_container_width=True,hide_index=True,height=440)
        else:
            st.info('No records selected.')

with tab_export:
    st.subheader('Spotfire-ready merged datasets')
    st.write('Selected runs are **stacked into one dataset** with `Case`, `Keyword`, '
             '`Column_Name`, `Entity`, `Unit` and simulation time. No renaming of ECLIPSE keywords is forced.')
    exp_level=st.radio('Data scope',levels+['All'],horizontal=True,key='export_level')
    submeta=meta_all if exp_level=='All' else meta_all.loc[meta_all.Level.eq(exp_level)]
    choices=sorted(submeta.Keyword.unique().tolist())
    ex_defaults=['FOPR','FWPR','FGPR','FOPT','FWPT','FGPT','FWCT','FPR',
                 'WOPR','WWPR','WGPR','WOPT','WWPT','WGPT','WWCT','WBHP','WTHP']
    export_all=st.checkbox(f'Include all {len(choices)} variables',value=False)
    export_kw=st.multiselect('Keyword / full name',choices,
                             default=choices if export_all else [x for x in ex_defaults if x in choices],
                             format_func=keyword_label,key=f'export_kw_{exp_level}_{export_all}')
    export_wells=[]
    if exp_level in ('Well','All'):
        export_wells=st.multiselect('Optional well filter · empty = all wells',all_wells,default=[])
    long_header_preview='Case,Date,Time_Days,Years,Level,Entity,Variable,Keyword,Column_Name,Unit,Value'
    st.code(long_header_preview+'\nCase_A,,365,0.9993,Field,FIELD,FOPR,FOPR,Field Oil Production Rate,STB/day,1000',
            language='csv')
    descriptive_wide=st.checkbox('Wide CSV: include full name in column headers',value=False,
                                  help='Off = compact ECLIPSE keyword headers. On = FOPR — Field Oil Production Rate.')
    selection={}
    series_count=0
    estimated_rows=0
    for case,parsed in cases.items():
        m=metadata_filter(parsed,exp_level,export_kw)
        if export_wells:
            m=m.loc[(m.Level!='Well')|m.Entity.isin(export_wells)]
        selection[case]=m
        series_count+=len(m)
        estimated_rows+=len(parsed['data'])*len(m)
    st.caption(f'{len(cases)} cases · {series_count:,} vectors · up to {estimated_rows:,} Long CSV rows before null filtering.')
    if estimated_rows>2_500_000:
        st.warning('Large dataset. Streamlit Cloud memory can be limited. Reduce keyword or well selections if needed.')
    export_zip=st.checkbox('Combine Long, Wide and Dictionary CSV in a ZIP',value=True)
    if st.button('Generate Spotfire exports',type='primary',disabled=series_count==0):
        try:
            with st.spinner('Building all-case CSV files...'):
                long_bytes=make_long_csv(cases,selection,unit_system,gas_display,start_date)
                wide_bytes=make_wide_csv(cases,selection,unit_system,gas_display,start_date,
                                         descriptive_headers=descriptive_wide)
                dict_bytes=make_dictionary_csv(cases,selection,unit_system,gas_display)
                zbytes=b''
                if export_zip:
                    memory=io.BytesIO()
                    with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                        z.writestr('RSM_Spotfire_LONG.csv',long_bytes)
                        z.writestr('RSM_Spotfire_WIDE.csv',wide_bytes)
                        z.writestr('RSM_Data_Dictionary.csv',dict_bytes)
                    zbytes=memory.getvalue()
                st.session_state['export_payload_v2']={
                    'long':long_bytes,'wide':wide_bytes,'dictionary':dict_bytes,'zip':zbytes,
                    'description':f'{len(cases)} cases · {series_count:,} vectors · {unit_system}',
                }
        except Exception as exc:
            st.session_state.pop('export_payload_v2',None)
            st.error(f'CSV generation failed: {exc}')
    payload=st.session_state.get('export_payload_v2')
    if payload:
        st.success('Generated: '+payload['description'])
        d1,d2,d3,d4=st.columns(4)
        with d1:
            st.download_button('Long CSV',payload['long'],file_name='RSM_Spotfire_LONG.csv',mime='text/csv')
        with d2:
            st.download_button('Wide CSV',payload['wide'],file_name='RSM_Spotfire_WIDE.csv',mime='text/csv')
        with d3:
            st.download_button('Data Dictionary',payload['dictionary'],file_name='RSM_Data_Dictionary.csv',mime='text/csv')
        with d4:
            if payload['zip']:
                st.download_button('All three — ZIP',payload['zip'],file_name='RSM_Spotfire_Exports.zip',
                                   mime='application/zip')
        st.caption('Re-generate after changing units, cases or export filters.')

with tab_qc:
    st.subheader('ECLIPSE run quality checks')
    qc=[]
    for case,parsed in cases.items():
        q=parsed['qc']
        qc.append({'Case':case,'RSM Filename':original_names[case],
                   'Blocks':q['Blocks'],'Timesteps':q['Time_Steps'],
                   'Vectors':q['Columns'],'Keywords':q['Unique_Keywords'],
                   'Wells':q['Wells'],'Scaled vectors':q['Scaled_Columns'],
                   'Mismatched time blocks':q['Time_Mismatched_Blocks'],
                   'Invalid numeric cells':q['NonNumeric_Data_Cells'],
                   'Start day':q['Time_Min_Days'],'End day':q['Time_Max_Days']})
    st.dataframe(pd.DataFrame(qc),use_container_width=True,hide_index=True)
    st.subheader('Field production versus summed wells')
    reconc=[]
    for case,parsed in cases.items():
        for fk,wk in [('FOPR','WOPR'),('FWPR','WWPR'),('FGPR','WGPR'),
                      ('FOPT','WOPT'),('FWPT','WWPT'),('FGPT','WGPT')]:
            fm=metadata_filter(parsed,'Field',[fk])
            wm=metadata_filter(parsed,'Well',[wk])
            if fm.empty or wm.empty:
                continue
            f,unit=value_and_unit(parsed['data'][fm.iloc[0].Column_ID].to_numpy(),
                                  fm.iloc[0],unit_system,gas_display)
            if not np.isfinite(f[-1]):
                continue
            sum_wells=0.0
            for r in wm.itertuples(index=False):
                w,_=value_and_unit(parsed['data'][r.Column_ID].to_numpy(),r._asdict(),
                                   unit_system,gas_display)
                if np.isfinite(w[-1]):
                    sum_wells+=float(w[-1])
            pct=100*(sum_wells-float(f[-1]))/abs(float(f[-1])) if f[-1] else np.nan
            reconc.append({'Case':case,'Field keyword':fk,'Well keyword':wk,
                            'Column_Name':keyword_description(fk),
                            'Field total':float(f[-1]),'Sum of wells':sum_wells,
                            'Difference %':pct,'Unit':unit})
    if reconc:
        st.dataframe(pd.DataFrame(reconc),use_container_width=True,hide_index=True)
        st.caption('Minor rounding differences are normal; missing well vectors can cause material gaps.')
    st.subheader('Keyword dictionary · RSM source scaling')
    dic=[]
    unsupported=[]
    for case,parsed in cases.items():
        m=parsed['metadata'].copy()
        m.insert(0,'Case',case)
        m.insert(m.columns.get_loc('Keyword')+1,'Column_Name',m.Keyword.map(keyword_description))
        m['Export_Unit']=[_target_conversion(r.Keyword,r.Source_Unit,unit_system,gas_display)[1]
                           for r in m.itertuples(index=False)]
        for r in m.itertuples(index=False):
            _,_,ok=_target_conversion(r.Keyword,r.Source_Unit,unit_system,gas_display)
            if not ok:
                unsupported.append((r.Keyword,r.Source_Unit))
        dic.append(m)
        if parsed['qc']['Issues']:
            st.warning(f"{case}: "+'; '.join(parsed['qc']['Issues'][:10]))
    st.dataframe(pd.concat(dic,ignore_index=True),use_container_width=True,hide_index=True,height=410)
    if unsupported:
        st.warning('Unit conversions not recognized — values preserved: '+
                   ', '.join(f'{k} [{u}]' for k,u in sorted(set(unsupported))[:25]))
    st.caption('Original RSM scaling and source units are preserved as metadata. Unknown keywords receive neutral labels.')
