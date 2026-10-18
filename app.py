"""Eclipse RSM Studio — upload, compare, visualize and Spotfire export .RSM runs.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rsm_parser import (
    make_dictionary_csv, make_long_csv, make_wide_csv,
    metadata_filter, parse_rsm, value_and_unit, _target_conversion,
)

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
