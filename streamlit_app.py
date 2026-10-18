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

# ---- Streamlit user interface ----
"""Eclipse RSM Studio — upload, compare, visualize and Spotfire export .RSM runs.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py
"""

from datetime import date
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Parser and conversion helpers are embedded above (single-file edition).

st.set_page_config(page_title="Eclipse RSM Studio", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 1.45rem; max-width: 1750px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    [data-testid="stMetricValue"] {font-size: 1.75rem; font-weight: 700;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
    .rsm-subtitle {color: #667085; font-size: 1rem; margin-top: -0.7rem; margin-bottom: 1rem;}
    .rsm-note {color: #667085; font-size: .82rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_parse(raw: bytes, filename: str) -> dict:
    return parse_rsm(raw, filename)


def to_x(times: np.ndarray, mode: str, start_date):
    if mode == "Years":
        return times / 365.25
    if mode == "Date" and start_date is not None:
        return pd.Timestamp(start_date) + pd.to_timedelta(times, unit="D")
    return times


def x_label(mode: str) -> str:
    return {"Time (days)": "Simulation time (days)",
            "Years": "Simulation time (years)", "Date": "Date"}.get(mode, "Simulation time")


def trace_data(parsed: dict, keyword: str, level: str, entity: str,
               system: str, gas_display: str):
    m = metadata_filter(parsed, level, [keyword], [entity])
    if m.empty:
        return None
    # Identical vector+entity+qualifier may theoretically recur; pick the first.
    r = m.iloc[0]
    values, unit = value_and_unit(parsed["data"][r.Column_ID].to_numpy(),
                                  r, system, gas_display)
    return parsed["data"].index.to_numpy(dtype=float), values, unit


def make_line_chart(cases: dict, level: str, keyword: str, entities: list[str],
                    system: str, gas_display: str, mode: str, start_date,
                    pct_watercut: bool = True, max_lines: int = 100,
                    height: int = 430) -> tuple[go.Figure, int]:
    fig = go.Figure()
    unit_name = ""
    traces = 0
    for case, parsed in cases.items():
        for entity in entities:
            if traces >= max_lines:
                break
            ser = trace_data(parsed, keyword, level, entity, system, gas_display)
            if ser is None:
                continue
            days, y, unit = ser
            if keyword.endswith("WCT") and pct_watercut:
                y = y * 100
                unit = "%"
            unit_name = unit
            name = f"{case} · {entity}" if level != "Field" else case
            fig.add_trace(go.Scattergl(
                x=to_x(days, mode, start_date), y=y,
                mode="lines", name=name, connectgaps=False,
                hovertemplate=f"{name}<br>{x_label(mode)}: %{{x}}<br>Value: %{{y:,.4g}} {unit}<extra></extra>",
                line=dict(width=2),
            ))
            traces += 1
    fig.update_layout(
        title=dict(text=f"{keyword} · {level}", font=dict(size=18)),
        height=height, template="plotly_white",
        xaxis_title=x_label(mode), yaxis_title=f"{keyword} ({unit_name})",
        margin=dict(t=65, b=45, l=58, r=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title=None),
        hovermode="x unified" if level == "Field" else "closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", rangemode="tozero")
    return fig, traces


def make_small_csv_preview(cases: dict, level: str, keywords: list[str],
                           entities: list[str], system: str, gas_display: str,
                           start_date, limit: int = 3000) -> pd.DataFrame:
    pieces = []
    left = limit
    for case, parsed in cases.items():
        m = metadata_filter(parsed, level, keywords,
                            entities if level != "Field" else ["FIELD"])
        td = parsed["data"].index.to_numpy(dtype=float)
        for r in m.itertuples(index=False):
            vals, unit = value_and_unit(parsed["data"][r.Column_ID].to_numpy(),
                                        r._asdict(), system, gas_display)
            n = min(left, len(td))
            if n == 0:
                break
            df = pd.DataFrame({
                "Case": case, "Time_Days": td[:n], "Years": td[:n] / 365.25,
                "Level": r.Level, "Entity": r.Entity, "Keyword": r.Keyword,
                "Variable": r.Variable, "Unit": unit, "Value": vals[:n],
            })
            if start_date is not None:
                df.insert(1, "Date", pd.Timestamp(start_date) + pd.to_timedelta(td[:n], unit="D"))
            pieces.append(df)
            left -= n
        if left <= 0:
            break
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


st.title("Eclipse RSM Studio")
st.markdown('<div class="rsm-subtitle">ECLIPSE summary analytics · Multi-case comparison · Metric / Field units · Spotfire-ready CSV</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 · Input cases")
    uploads = st.file_uploader(
        "Upload Eclipse .RSM files", type=["rsm", "txt"],
        accept_multiple_files=True,
        help="Upload multiple simulations together. Every file is kept as a separate named case."
    )
    st.header("2 · Unit settings")
    unit_system = st.radio("Output units", ["Metric", "Field"], horizontal=True)
    gas_display = "Mscf"
    if unit_system == "Field":
        gas_display = st.selectbox("Gas volume display", ["Mscf", "MMscf"])
    st.header("3 · Time axis")
    use_calendar = st.checkbox("Provide simulation start date", value=False,
                               help="RSM contains elapsed days, not necessarily a calendar start date.")
    start_date = st.date_input("Simulation DAY 0", value=date.today()) if use_calendar else None
    axis_modes = ["Time (days)", "Years"] + (["Date"] if use_calendar else [])
    x_mode = st.radio("Chart X axis", axis_modes, horizontal=True)
    st.caption("Calendar dates are calculated only if you provide the correct model start date.")

if not uploads:
    st.info("Upload one or more Eclipse ASCII **.RSM** files using the sidebar to begin.")
    st.markdown("""
    **What you can do here**
    - Compare different run cases side-by-side using original RSM filenames or custom labels.
    - View **Field**, **Well**, and **Group** vector histories and ranked wells.
    - Toggle between **Metric** and **Field** engineering units.
    - Download **Long CSV**, **Wide CSV** and **Data Dictionary** for Spotfire.

    This reader is designed for text-based **Eclipse RSM** exports, not binary `UNSMRY` or `SMSPEC` files.
    """)
    st.stop()

cases = {}
original_names = {}
errors = []
with st.sidebar:
    st.header("4 · Name the cases")
    for i, uploaded in enumerate(uploads):
        default = Path(uploaded.name).stem
        label = st.text_input(f"Case {i+1} · {uploaded.name}", value=default,
                              key=f"case_{i}_{uploaded.name}").strip()
        label = label or default
        if label in cases:
            label = f"{label} ({i+1})"
        try:
            with st.spinner(f"Parsing {uploaded.name}..."):
                cases[label] = cached_parse(uploaded.getvalue(), uploaded.name)
                original_names[label] = uploaded.name
        except Exception as exc:
            errors.append(f"{uploaded.name}: {exc}")

if errors:
    for err in errors:
        st.error(err)
if not cases:
    st.stop()

meta_all = pd.concat([x["metadata"] for x in cases.values()], ignore_index=True)
all_wells = sorted(meta_all.loc[meta_all.Level.eq("Well"), "Entity"].unique().tolist())
all_groups = sorted(meta_all.loc[meta_all.Level.eq("Group"), "Entity"].unique().tolist())

s1, s2, s3, s4 = st.columns(4)
s1.metric("Cases loaded", f"{len(cases)}")
s2.metric("Unique wells", f"{len(all_wells)}")
s3.metric("Summary keywords", f"{meta_all.Keyword.nunique()}")
s4.metric("Longest run", f"{max(c['qc']['Time_Max_Days'] for c in cases.values()):,.0f} days")

if any(x["qc"]["Time_Mismatched_Blocks"] for x in cases.values()):
    st.warning("One or more RSM blocks have differing TIME vectors. Blocks were merged by their numeric time values (outer join).")

_tab_overview, _tab_charts, _tab_rank, _tab_data, _tab_export, _tab_qc = st.tabs([
    "Overview", "Field & well plots", "Well ranking", "Data explorer", "Spotfire CSV export", "QC & metadata"
])

with _tab_overview:
    st.subheader("Field summary · last available timestep")
    main_vectors = ["FOPR", "FWPR", "FGPR", "FOPT", "FWPT", "FGPT", "FWCT", "FPR"]
    summary = []
    for case, parsed in cases.items():
        rec = {"Case": case, "End (days)": parsed["qc"]["Time_Max_Days"]}
        for keyword in main_vectors:
            info = trace_data(parsed, keyword, "Field", "FIELD", unit_system, gas_display)
            if info is None:
                continue
            _, vals, unit = info
            finite = vals[np.isfinite(vals)]
            rec[f"{keyword} [{unit}]"] = float(finite[-1]) if len(finite) else np.nan
        summary.append(rec)
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.caption("These are **end-of-run values**, not sums across all dates. Cumulative vectors include each RSM column's printed scale factor.")

    st.subheader("Field production dashboard")
    has_fields = {s for s in meta_all.loc[meta_all.Level.eq("Field"), "Keyword"].unique()}
    overview_pairs = [("FOPR", "Oil production rate"), ("FWPR", "Water production rate"),
                      ("FGPR", "Gas production rate"), ("FOPT", "Cumulative oil"),
                      ("FWCT", "Water cut"), ("FPR", "Reservoir pressure")]
    for j in range(0, len(overview_pairs), 2):
        cols = st.columns(2)
        for col, (key, title) in zip(cols, overview_pairs[j:j+2]):
            if key not in has_fields:
                continue
            with col:
                fig, n = make_line_chart(cases, "Field", key, ["FIELD"],
                                         unit_system, gas_display, x_mode, start_date,
                                         height=315)
                if n:
                    fig.update_layout(title=title, margin=dict(t=68,b=45,l=58,r=10))
                    st.plotly_chart(fig, use_container_width=True, key=f"ov_{key}")

with _tab_charts:
    a, b, c = st.columns([1.1, 1.4, 1.8])
    levels_found = [v for v in ("Field", "Well", "Group") if v in set(meta_all.Level)]
    with a:
        chart_level = st.selectbox("Analysis level", levels_found, key="chart_level")
    choices = sorted(meta_all.loc[meta_all.Level.eq(chart_level), "Keyword"].unique().tolist())
    default_key = {"Field": "FOPR", "Well": "WOPR", "Group": "GOPR"}.get(chart_level)
    with b:
        chart_key = st.selectbox("Eclipse keyword", choices,
                                 index=choices.index(default_key) if default_key in choices else 0,
                                 key=f"chart_var_{chart_level}")
    with c:
        chart_cases = st.multiselect("Cases", list(cases), default=list(cases), key="chart_cases")

    chart_entities = ["FIELD"]
    if chart_level in ("Well", "Group"):
        chart_entities_available = sorted(meta_all.loc[
            meta_all.Level.eq(chart_level) & meta_all.Keyword.eq(chart_key), "Entity"
        ].unique().tolist())
        chart_entities = st.multiselect(
            f"{chart_level}s", chart_entities_available,
            default=chart_entities_available[:4] if chart_level == "Well" else chart_entities_available,
            key=f"entities_{chart_level}_{chart_key}"
        )
    pct_water = st.checkbox("Plot water cut as % (CSV retains fraction)", value=True)
    chart_subset = {n: cases[n] for n in chart_cases}
    if chart_subset and chart_entities:
        fig, n = make_line_chart(chart_subset, chart_level, chart_key, chart_entities,
                                 unit_system, gas_display, x_mode, start_date, pct_water)
        if n:
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
            if n >= 100:
                st.warning("Displayed the first 100 curves to protect browser performance; narrow your well selection to view more.")
            latest = []
            for name, parsed in chart_subset.items():
                for entity in chart_entities:
                    data = trace_data(parsed, chart_key, chart_level, entity,
                                      unit_system, gas_display)
                    if data is None:
                        continue
                    td, v, unit = data
                    finite_idx = np.where(np.isfinite(v))[0]
                    if not len(finite_idx):
                        continue
                    j = finite_idx[-1]
                    latest.append({"Case": name, "Level": chart_level, "Entity": entity,
                                   "Keyword": chart_key, "Last_Time_Days": td[j],
                                   "Last_Value": v[j], "Unit": unit})
            st.markdown("**Last recorded value for each selected curve**")
            st.dataframe(pd.DataFrame(latest), use_container_width=True, hide_index=True)
        else:
            st.info("No matching vectors for the selected cases/wells.")
    else:
        st.info("Choose at least one case and one entity to plot.")

with _tab_rank:
    st.subheader("Well ranking at a selected simulation time")
    well_keywords = sorted(meta_all.loc[meta_all.Level.eq("Well"), "Keyword"].unique().tolist())
    rank_col1, rank_col2, rank_col3 = st.columns([1, 1, 1])
    with rank_col1:
        rank_key = st.selectbox("Ranking variable", well_keywords,
                                index=well_keywords.index("WOPR") if "WOPR" in well_keywords else 0)
    max_time = max(v["qc"]["Time_Max_Days"] for v in cases.values())
    min_time = min(v["qc"]["Time_Min_Days"] for v in cases.values())
    with rank_col2:
        rank_day = st.slider("Simulation day", min_value=float(min_time),
                             max_value=float(max_time), value=float(max_time),
                             step=max(1.0, round(float(max_time - min_time) / 900)))
    with rank_col3:
        top_n = st.number_input("Top wells / case", min_value=1, max_value=max(1, len(all_wells)),
                                value=min(20, max(1, len(all_wells))))
    rows = []
    for case, parsed in cases.items():
        timeline = parsed["data"].index.to_numpy(dtype=float)
        pick = int(np.abs(timeline - rank_day).argmin())
        m = metadata_filter(parsed, "Well", [rank_key])
        for r in m.itertuples(index=False):
            v, unit = value_and_unit(parsed["data"][r.Column_ID].to_numpy(),
                                     r._asdict(), unit_system, gas_display)
            if np.isfinite(v[pick]):
                rows.append({"Case": case, "Well": r.Entity, "Time_Days": float(timeline[pick]),
                             "Keyword": rank_key, "Value": float(v[pick]), "Unit": unit})
    if rows:
        rank = pd.DataFrame(rows)
        rank = (rank.sort_values("Value", ascending=False)
                .groupby("Case", group_keys=False).head(int(top_n)))
        fig = px.bar(rank, x="Well", y="Value", color="Case", barmode="group",
                     title=f"{rank_key} · Top {top_n} wells per case",
                     labels={"Value": f"{rank_key} ({rank.iloc[0].Unit})"},
                     template="plotly_white", hover_data=["Time_Days"])
        fig.update_layout(height=490, xaxis_tickangle=-45,
                          margin=dict(t=60,b=85,l=50,r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rank, use_container_width=True, hide_index=True)
        st.download_button("Download well ranking CSV", rank.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"well_ranking_{rank_key}.csv", mime="text/csv")
    else:
        st.info("No well values are available at this time.")

with _tab_data:
    st.subheader("Tabular data explorer")
    a, b = st.columns([1, 2])
    with a:
        preview_level = st.selectbox("Level", [v for v in ("Field", "Well", "Group") if v in set(meta_all.Level)], key="expl_level")
    available = sorted(meta_all.loc[meta_all.Level.eq(preview_level), "Keyword"].unique().tolist())
    preferred = {"Field": ["FOPR", "FWPR", "FOPT"],
                 "Well": ["WOPR", "WWPR", "WOPT"],
                 "Group": ["GOPR", "GWPR"]}.get(preview_level, [])
    with b:
        preview_keywords = st.multiselect("Variables", available,
                                          default=[x for x in preferred if x in available],
                                          key=f"preview_k_{preview_level}")
    if preview_level == "Field":
        preview_entities = ["FIELD"]
    else:
        options = sorted(meta_all.loc[meta_all.Level.eq(preview_level), "Entity"].unique().tolist())
        preview_entities = st.multiselect("Entities", options,
                                          default=options[:3], key=f"preview_ent_{preview_level}")
    st.caption("Preview is capped at 3,000 rows. CSV export below can include the complete selected time series.")
    if preview_keywords and preview_entities:
        table = make_small_csv_preview(cases, preview_level, preview_keywords,
                                       preview_entities, unit_system, gas_display,
                                       start_date, 3000)
        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True, height=460)
        else:
            st.info("No matching data.")

with _tab_export:
    st.subheader("Spotfire-ready merged datasets")
    st.write("All selected cases will be **stacked into one CSV** with a `Case` column. "
             "Use Long format for flexible Spotfire analysis and Wide format for one row per case/time/entity.")
    st.markdown("**Example — Long CSV (recommended):**")
    st.code("Case,Date,Time_Days,Years,Level,Entity,Variable,Keyword,Unit,Value\n"
            "Case_A,,365,0.9993,Well,RU-019,WOPR,WOPR,STB/day,500\n"
            "Case_B,,365,0.9993,Well,RU-019,WOPR,WOPR,STB/day,470", language="csv")
    exp_level = st.radio("Export scope", [v for v in ("Field", "Well", "Group") if v in set(meta_all.Level)] + ["All"], horizontal=True)
    possible = meta_all if exp_level == "All" else meta_all.loc[meta_all.Level.eq(exp_level)]
    export_keywords = sorted(possible.Keyword.unique().tolist())
    defaults = ["FOPR", "FWPR", "FGPR", "FOPT", "FWPT", "FGPT", "FWCT", "FPR",
                "WOPR", "WWPR", "WGPR", "WOPT", "WWPT", "WGPT", "WWCT", "WBHP", "WTHP", "WGOR"]
    use_all = st.checkbox(f"Select all {len(export_keywords)} keywords", value=False)
    chosen_kw = st.multiselect("Select keywords", export_keywords,
                               default=export_keywords if use_all else [x for x in defaults if x in export_keywords],
                               key=f"exkw_{exp_level}_{use_all}")
    choose_wells = []
    if exp_level in ("Well", "All"):
        choose_wells = st.multiselect("Optional well filter (empty = all wells)", all_wells,
                                      default=[], key="exp_wells")
    selection = {}
    n_cols = 0
    expected_rows = 0
    for name, parsed in cases.items():
        m = metadata_filter(parsed, exp_level, chosen_kw)
        if choose_wells:
            m = m.loc[(m.Level != "Well") | m.Entity.isin(choose_wells)]
        selection[name] = m
        n_cols += len(m)
        expected_rows += len(parsed["data"]) * len(m)
    st.caption(f"Selection: {len(cases)} cases · {n_cols:,} series · approximately {expected_rows:,} Long CSV rows (before null filtering).")
    if expected_rows > 2_500_000:
        st.warning("Large export. Reduce variables/wells if browser memory or Streamlit Cloud resources are limited.")

    combined_zip = st.checkbox("Also package all 3 CSVs in one ZIP", value=True)
    if st.button("Generate Spotfire export", type="primary", disabled=n_cols == 0):
        with st.spinner("Writing Spotfire Long, Wide and Dictionary files..."):
            try:
                long_bytes = make_long_csv(cases, selection, unit_system, gas_display, start_date)
                wide_bytes = make_wide_csv(cases, selection, unit_system, gas_display, start_date)
                dict_bytes = make_dictionary_csv(cases, selection, unit_system, gas_display)
                zip_bytes = b""
                if combined_zip:
                    mem = io.BytesIO()
                    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED,
                                         compresslevel=6) as z:
                        z.writestr("RSM_Spotfire_LONG.csv", long_bytes)
                        z.writestr("RSM_Spotfire_WIDE.csv", wide_bytes)
                        z.writestr("RSM_Data_Dictionary.csv", dict_bytes)
                    zip_bytes = mem.getvalue()
                st.session_state["export_payload"] = {
                    "long": long_bytes, "wide": wide_bytes, "dictionary": dict_bytes,
                    "zip": zip_bytes, "config": f"{len(cases)} cases · {n_cols} series · {unit_system}"
                }
            except Exception as ex:
                st.error(f"Export failed: {ex}")
                st.session_state.pop("export_payload", None)
    payload = st.session_state.get("export_payload")
    if payload:
        st.success(f"CSV files generated: {payload['config']}.")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("Download Long CSV", payload["long"],
                               file_name="RSM_Spotfire_LONG.csv", mime="text/csv")
        with d2:
            st.download_button("Download Wide CSV", payload["wide"],
                               file_name="RSM_Spotfire_WIDE.csv", mime="text/csv")
        with d3:
            st.download_button("Download Dictionary", payload["dictionary"],
                               file_name="RSM_Data_Dictionary.csv", mime="text/csv")
        with d4:
            if payload["zip"]:
                st.download_button("Download all as ZIP", payload["zip"],
                                   file_name="RSM_Spotfire_Exports.zip", mime="application/zip")
        st.caption("Downloaded files reflect the selections used when you last clicked Generate. Click Generate again after changing filters.")

with _tab_qc:
    st.subheader("Quality checks for each run")
    qc = []
    for case, parsed in cases.items():
        q = parsed["qc"]
        qc.append({"Case": case, "Filename": original_names[case], "Blocks": q["Blocks"],
                   "TimeSteps": q["Time_Steps"], "Vectors": q["Columns"],
                   "Keywords": q["Unique_Keywords"], "Wells": q["Wells"],
                   "ScaledColumns": q["Scaled_Columns"],
                   "MisalignedBlocks": q["Time_Mismatched_Blocks"],
                   "NonNumericCells": q["NonNumeric_Data_Cells"],
                   "StartDays": q["Time_Min_Days"], "EndDays": q["Time_Max_Days"]})
    st.dataframe(pd.DataFrame(qc), use_container_width=True, hide_index=True)
    st.caption("NonNumericCells counts non-empty values that could not be interpreted as numbers; blank cells remain missing, not zero.")

    st.subheader("Field versus sum of wells · end-of-run reconciliation")
    st.caption("Independent QC: compare simulator field totals against the same variables summed for all exported wells at each case's last simulation day.")
    recon = []
    counterparts = [("FOPR", "WOPR"), ("FWPR", "WWPR"),
                    ("FGPR", "WGPR"), ("FOPT", "WOPT"),
                    ("FWPT", "WWPT"), ("FGPT", "WGPT")]
    for case, parsed in cases.items():
        for field_key, well_key in counterparts:
            fm = metadata_filter(parsed, "Field", [field_key])
            wm = metadata_filter(parsed, "Well", [well_key])
            if fm.empty or wm.empty:
                continue
            f, u = value_and_unit(parsed["data"][fm.iloc[0].Column_ID].to_numpy(),
                                  fm.iloc[0], unit_system, gas_display)
            field_value = float(f[-1])
            well_total = 0.0
            for r in wm.itertuples(index=False):
                v, _ = value_and_unit(parsed["data"][r.Column_ID].to_numpy(),
                                      r._asdict(), unit_system, gas_display)
                if np.isfinite(v[-1]):
                    well_total += float(v[-1])
            diff_pct = 100.0 * (well_total - field_value) / abs(field_value) if field_value else np.nan
            recon.append({"Case": case, "Field_Vector": field_key,
                          "Well_Vector": well_key, "Field_Value": field_value,
                          "Sum_Wells": well_total, "Unit": u,
                          "Difference_%": diff_pct})
    if recon:
        st.dataframe(pd.DataFrame(recon), use_container_width=True, hide_index=True)
        st.caption("Minor discrepancies are expected due to printed RSM rounding; material gaps may indicate omitted/inconsistent summary vectors.")
    st.subheader("Variable dictionary and source scale")
    meta_sample = []
    unsupported = set()
    for name, parsed in cases.items():
        m = parsed["metadata"].copy()
        m.insert(0, "Case", name)
        m["Export_Unit"] = [
            _target_conversion(r.Keyword, r.Source_Unit, unit_system, gas_display)[1]
            for r in m.itertuples(index=False)
        ]
        for r in m.itertuples(index=False):
            _, _, supported = _target_conversion(r.Keyword, r.Source_Unit,
                                                 unit_system, gas_display)
            if not supported:
                unsupported.add((r.Keyword, r.Source_Unit))
        meta_sample.append(m)
    metadata_out = pd.concat(meta_sample, ignore_index=True)
    st.dataframe(metadata_out, use_container_width=True, hide_index=True, height=380)
    if unsupported:
        st.warning("Unrecognized or dimensionless source units are preserved without forced conversion: " +
                   ", ".join(f"{kw} [{unit or 'blank'}]" for kw, unit in sorted(unsupported)[:25]))
    else:
        st.success("All recognized vectors have supported unit metadata or a dimensionless type.")
    for case, parsed in cases.items():
        if parsed["qc"]["Issues"]:
            st.warning(f"{case}: " + "; ".join(parsed["qc"]["Issues"][:20]))
    st.caption("NOTE: RSM column scaling and unit conversions are applied to displayed/exported values; original unscaled numbers remain available in memory for reliable re-conversion.")
