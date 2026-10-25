# Eclipse RSM Studio Pro — Extended Version

## What's new

The existing Eclipse RSM parser, Field Overview dashboard, engineering unit conversions, interactive plots/Trellis, well ranking, data explorer, Spotfire export and QC remain in place.

1. **Save / load design template**
   - Open sidebar section `0 · Save / load design template` to upload a previously downloaded `.json` configuration.
   - After customizing displays and filters, use the `Download design template (.json)` button at the bottom of the sidebar.
   - The JSON contains case labels (matched to original uploaded RSM file names), field/metric units, dates, x-axis selection, color/font/line options, well profile selection, plot/Trellis layout, explorer and export selections, and optional STOIIP overrides.
   - **RSM data files are NOT embedded.** Re-upload your original `.RSM` files whenever reopening a session. Missing files are flagged.
   - Options no longer present in uploaded files are dropped from restored selections; filters are not silently applied to non-existent cases or wells.

2. **Well by Well Profiles**
   - Choose multiple wells and one or more cases and color by **Well**, **Case**, or **Keyword**.
   - Side-by-side dashboards automatically show available standard vectors such as WOPR, WWPR, WGPR, WOPT, WWCT, WBHP, WTHP and WWPT.
   - Download final well-level data snapshot as CSV.

3. **Volumes & Recovery**
   - Report **initial STOIIP**, **recovered oil** and **recovery factor (RF)** for each case.
   - Initial STOIIP: first valid `FOIP` value or, when FOIP is missing, estimated from `FOPT / FOE` (with both variables measured at the same timestep). Optional manual override is available per case and engineering unit system.
   - Recovered oil: final valid `FOPT` (cumulative production).
   - RF (%): `100 × FOPT / initial STOIIP`; if STOIIP cannot be estimated, display the reported `FOE × 100` instead.
   - The tab includes a case comparison table, KPI cards, initial STOIIP bar chart, FOPT profiles, RF profiles, and remaining FOIP curves where provided.
   - All calculations use ECLIPSE's reported scaling and unit conversions. No missing input is assumed to equal zero.
   - If the first FOIP timestep occurs later than simulation day zero, the app warns that this might not equal the true initial STOIIP. Validate against independently established STOIIP values. Using different ECLIPSE restart dates/scenarios can affect comparison.

## Deploy to GitHub / Streamlit Community Cloud

1. Upload **`streamlit_app.py`** and **`requirements.txt`** to the top level of a GitHub repository (do not put them inside an extra folder).
2. Select `streamlit_app.py` as the main file in Streamlit Cloud.
3. Open the app; upload your Eclipse ASCII `.RSM` runs.
4. Customize the display, then download a `.json` design template. At the next visit, re-upload the `.RSM` files and your `.json` design file to restore selections.

No external credentials are required. Large runs can have long render times; the existing per-chart cap applies and CSV export is not capped by the plot limit.
