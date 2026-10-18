# Eclipse RSM Studio

Streamlit application for reviewing **Eclipse ASCII Summary (.RSM)** output, comparing multiple runs and exporting Spotfire-friendly tables.

## Quick start — GitHub + Streamlit Community Cloud

1. **Easiest deployment:** unzip and upload just **`streamlit_app.py`** and **`requirements.txt`** into the root of a GitHub repository (two normal files, no folders needed). The remaining files are an optional modular edition, tests and documentation.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select the repository and branch, and set the **Main file path** to `streamlit_app.py`.
3. Deploy, then use the sidebar to upload one or more `.RSM` files.
4. Edit each case name as needed. Each run remains a separate `Case` in exports.
5. Select `Metric` or `Field`, view field- and well-level plots, and download Spotfire CSV files.

Local execution:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run unit tests (optional):

```bash
python -m pip install pytest
python -m pytest -q
```

## Supported workflow

- Upload and compare multiple Eclipse **ASCII** RSM summary files; file names become case labels by default.
- Read many side-by-side printed RSM sections/pages automatically, aligning columns on their `TIME` (days) axis.
- Read vector metadata (e.g. `FOPR`, `WOPR`, `FOPT`, `WOPT`, `WBHP`) and entity names separately. Supports **Field, Well, Group** vectors.
- Correct the Eclipse RSM `*10**3` etc. printed column scale for true engineering quantities.
- Unit system toggle: **Metric** (Sm3/day, Rm3, bara) / **Field** (STB/day, RB, psia, Mscf/day or MMscf/day).
- Plot time histories and multi-case comparisons, rank wells for any point in simulation time, inspect data, and QC metadata.
- Optional simulation start date: user must supply **true model DAY 0** to obtain calendar dates. When disabled, the Spotfire `Date` column is blank and `Time_Days`/`Years` are preserved.

## Exports

Click **Spotfire CSV export** and choose level, keyword(s), optional wells, and then **Generate Spotfire export**.

| File | Shape | Spotfire usage |
|---|---|---|
| `RSM_Spotfire_LONG.csv` | `Case, Date, Time_Days, Years, Level, Entity, Variable, Keyword, Unit, Value` | Filter `Keyword` and `Entity`; color charts by `Case` |
| `RSM_Spotfire_WIDE.csv` | One row per `Case + Time_Days + Level + Entity`; variables are columns | Easy direct cross-case plots and well-level calculations |
| `RSM_Data_Dictionary.csv` | Variable, original printed unit and scale, export unit | Validate conversion and interpret column metadata |
| `RSM_Spotfire_Exports.zip` | All three files together | Easier download for very large exports (unzip before Spotfire) |

Files have row-1 headers, no merged cells, numeric column types, and UTF-8 with BOM so they import cleanly into Spotfire/Excel. Missing values are empty cells, not artificial zeroes. Multiple cases need not have identical wells or time grids; records are **stacked**, not summed or interpolated.

## Unit notes

| Quantity | Metric | Field |
|---|---|---|
| Oil/water surface volume | Sm3 | STB (1 Sm3 = 6.28981077 STB) |
| Oil/water surface rate | Sm3/day | STB/day |
| Surface gas volume/rate | Sm3; Sm3/day | Mscf or MMscf; /day (1 Sm3 = 35.3146667 scf) |
| Reservoir fluid volume/rate | Rm3; Rm3/day | RB; RB/day |
| Absolute pressure | bara | psia (1 bar = 14.5037738 psi) |
| GOR (`FGOR`, `WGOR`, ...) | Sm3/Sm3 | scf/STB |
| Water/gas ratio (`FWGR`, `WWGR`, ...) | Sm3/Sm3 | STB/MMscf |
| Water cut (`FWCT`, `WWCT`) | fraction | fraction (plot can show %) |
| Oil PI | Sm3/day/bar | STB/day/psi |

Unrecognized/dimensionless source units are **preserved** rather than converted with a speculative rule. Check the **QC & metadata** tab after uploading each file.

**Important:** `.RSM` is a text summary export. This app does **not** read binary `UNSMRY` / `SMSPEC`, grid `EGRID`, restart `UNRST`, or the Eclipse run log `PRT` output. The app cannot recover missing vectors that were never written into RSM. If multiple scenarios share the same keyword but different wells, only matching well and case histories are compared, never aggregated silently. Simulation-specific meanings of exotic Eclipse keywords should be checked against the model's summary definition.

## Performance and privacy

The uploaded RSM is parsed in the running Streamlit process. The supplied sample run is ~13 MB and contains more than 1,000 time-series columns. Exports selecting all keywords and all wells may be tens/hundreds of MB; filter down if using Community Cloud. Uploaded model data is sent to the host running your Streamlit app. **Do not commit your confidential RSM files to public GitHub**; users upload simulation results within the app instead. Use a private/self-hosted deployment where your company policy requires it.

## Project structure

```text
streamlit_app.py             # Recommended single-file app (includes parser)
app.py                       # Alternative modular Streamlit UI
rsm_parser.py                # Parser for alternative modular app, conversions, exports
requirements.txt             # Python libraries
.streamlit/config.toml       # UI theme and upload size
README.md                    # This guide
tests/test_parser.py         # Parser/unit conversion/export tests
```
