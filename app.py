"""
EM2 Standards Alignment Explorer
Streamlit app for exploring Learnosity alignment data.

Tabs:
    1. CCSS → State Standards  — pick a CCSS standard, see all aligned state standards + map
    2. State Standards Browser  — pick a state, browse all its standards with CCSS matches
"""

import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── CONFIG ────────────────────────────────────────────────────────────────────

DB_PATH = "learnosity.db"

# All 50 states for the choropleth — must include states with no data so the
# full map renders rather than just the states that have alignments.
ALL_US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
    "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]


# ── PAGE SETUP ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EM2 Alignment Explorer",
    page_icon="📐",
    layout="wide",
)

if not os.path.exists(DB_PATH):
    st.error(
        f"**Database not found.** Expected `{DB_PATH}` in the app directory. "
        "Run `build_database.py` to generate it, then restart the app."
    )
    st.stop()


# ── DATABASE QUERIES ──────────────────────────────────────────────────────────
# @st.cache_data caches the return value so repeated calls (e.g. on every
# widget interaction) don't hit the database again.

@st.cache_data
def load_ccss_standards() -> pd.DataFrame:
    """All CCSS standards, ordered for display."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            """
            SELECT ccss_id, code, grade, domain, domain_name, text
            FROM   ccss_standards
            ORDER  BY grade, domain, standard, is_subpart
            """,
            conn,
        )


@st.cache_data
def load_states() -> pd.DataFrame:
    """All state / standards-year combinations."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            """
            SELECT state_id, abbreviation, standards_year
            FROM   states
            ORDER  BY abbreviation, standards_year
            """,
            conn,
        )


@st.cache_data
def load_ccss_alignments(ccss_id: int) -> pd.DataFrame:
    """All state standards aligned to a given CCSS standard."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            """
            SELECT
                s.abbreviation,
                s.standards_year,
                ss.standard_code,
                ss.grade          AS state_grade,
                ss.standard_text
            FROM  crosswalk        cw
            JOIN  state_standards  ss ON cw.state_standard_id = ss.state_standard_id
            JOIN  states           s  ON ss.state_id           = s.state_id
            WHERE cw.ccss_id = ?
            ORDER BY s.abbreviation, s.standards_year, ss.standard_code
            """,
            conn,
            params=(ccss_id,),
        )


@st.cache_data
def load_state_standards(state_id: int) -> pd.DataFrame:
    """
    All standards for a given state, with their CCSS alignments aggregated
    into a single comma-separated string. Standards with no alignment in the
    crosswalk still appear (LEFT JOIN), with NULL in ccss_alignments.
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            """
            SELECT
                ss.standard_code,
                ss.grade,
                ss.standard_text,
                GROUP_CONCAT(DISTINCT cs.code) AS ccss_alignments
            FROM  state_standards  ss
            LEFT  JOIN crosswalk       cw ON ss.state_standard_id = cw.state_standard_id
            LEFT  JOIN ccss_standards  cs ON cw.ccss_id            = cs.ccss_id
            WHERE ss.state_id = ?
            GROUP BY ss.state_standard_id
            ORDER BY
                (ss.grade IS NULL),                                  -- matched (graded) rows first
                CASE ss.grade WHEN 'K' THEN 0 ELSE CAST(ss.grade AS INTEGER) END,
                ss.standard_code
            """,
            conn,
            params=(state_id,),
        )


# ── HELPERS ───────────────────────────────────────────────────────────────────

def format_year(y) -> str:
    """Convert a standards_year value to a display string."""
    if pd.isna(y) or y == 0:
        return "—"
    return str(int(y))


def build_ccss_label(row: dict) -> str:
    """One-line label for the CCSS selectbox: grade tag + code + text preview."""
    text = str(row["text"]) if pd.notna(row["text"]) else ""
    preview = text[:80] + "…" if len(text) > 80 else text
    return f"[Gr {row['grade']}]  {row['code']}  —  {preview}"


def build_alignment_map(alignments_df: pd.DataFrame) -> go.Figure:
    """
    US choropleth coloured by number of aligned state standards per state.
    Unaligned states render in light gray; aligned states shade from light to
    dark teal proportional to their count.
    """
    counts = (
        alignments_df.groupby("abbreviation")
        .size()
        .reset_index(name="n")
    )
    count_map = dict(zip(counts["abbreviation"], counts["n"]))

    z_values = [count_map.get(s, 0) for s in ALL_US_STATES]
    hover_labels = [
        f"<b>{s}</b><br>{count_map[s]} aligned standard(s)"
        if s in count_map
        else f"<b>{s}</b><br>No alignment"
        for s in ALL_US_STATES
    ]

    fig = go.Figure(
        go.Choropleth(
            locations=ALL_US_STATES,
            z=z_values,
            locationmode="USA-states",
            colorscale=[[0, "#E8E8E8"], [1, "#1D9E75"]],
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.5,
            hovertext=hover_labels,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        geo_scope="usa",
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
    )
    return fig


# ── APP ───────────────────────────────────────────────────────────────────────

st.title("EM2 Standards Alignment Explorer")

tab1, tab2 = st.tabs(["CCSS → State Standards", "State Standards Browser"])


# ── TAB 1: CCSS → STATE STANDARDS ────────────────────────────────────────────

with tab1:
    ccss_df = load_ccss_standards()

    # Build the lookup structures the selectbox needs.
    records   = ccss_df.to_dict("records")
    labels    = [build_ccss_label(r) for r in records]
    id_by_lbl = dict(zip(labels, ccss_df["ccss_id"]))
    row_by_lbl = dict(zip(labels, records))

    chosen_label = st.selectbox(
        "Search for a CCSS standard",
        options=[""] + labels,
        format_func=lambda x: x or "— type a grade, code, or keyword to search —",
    )

    if chosen_label:
        ccss_id = id_by_lbl[chosen_label]
        std = row_by_lbl[chosen_label]

        # Standard detail card.
        with st.container(border=True):
            st.markdown(
                f"**{std['code']}** &nbsp; Grade {std['grade']} "
                f"&nbsp;·&nbsp; {std['domain_name']}"
            )
            st.write(std["text"] if pd.notna(std["text"]) else "*No text available.*")

        alignments = load_ccss_alignments(ccss_id)

        if alignments.empty:
            st.info("No state alignments found for this standard.")
        else:
            n_states = alignments["abbreviation"].nunique()
            st.caption(
                f"{len(alignments)} aligned state standard(s) "
                f"across {n_states} state(s)"
            )

            col_table, col_map = st.columns([3, 2])

            with col_table:
                display = alignments.rename(columns={
                    "abbreviation":   "State",
                    "standards_year": "Year",
                    "standard_code":  "Code",
                    "state_grade":    "Grade",
                    "standard_text":  "Text",
                })
                display["Year"] = display["Year"].apply(format_year)
                display = display.fillna("—")

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Text": st.column_config.TextColumn(width="large"),
                    },
                )

                st.download_button(
                    "Download as CSV",
                    data=display.to_csv(index=False).encode("utf-8"),
                    file_name=f"ccss_{std['code'].replace('.', '_')}_alignments.csv",
                    mime="text/csv",
                )

            with col_map:
                st.plotly_chart(
                    build_alignment_map(alignments),
                    use_container_width=True,
                )


# ── TAB 2: STATE STANDARDS BROWSER ───────────────────────────────────────────

with tab2:
    states_df  = load_states()
    abbrs      = sorted(states_df["abbreviation"].dropna().unique().tolist())

    selected_abbr = st.selectbox("Select a state", abbrs)

    # If the state has multiple standards-year versions, let the user pick one.
    versions = states_df[
        states_df["abbreviation"] == selected_abbr
    ].reset_index(drop=True)

    if len(versions) > 1:
        year_labels = [format_year(row["standards_year"]) for _, row in versions.iterrows()]
        chosen_year = st.selectbox("Standards version", year_labels)
        idx = year_labels.index(chosen_year)
        state_id = int(versions.iloc[idx]["state_id"])
    else:
        state_id = int(versions.iloc[0]["state_id"])

    standards = load_state_standards(state_id)

    search = st.text_input(
        "Filter",
        placeholder="Search by code or standard text…",
    )
    if search:
        mask = (
            standards["standard_code"].str.contains(search, case=False, na=False)
            | standards["standard_text"].str.contains(search, case=False, na=False)
        )
        standards = standards[mask]

    total     = len(standards)
    matched   = standards["ccss_alignments"].notna().sum()
    unmatched = total - matched

    st.caption(
        f"{total} standard(s) shown  ·  "
        f"{matched} with CCSS alignment  ·  "
        f"{unmatched} unmatched"
    )

    display = standards.rename(columns={
        "standard_code":   "Code",
        "grade":           "Grade",
        "standard_text":   "Text",
        "ccss_alignments": "CCSS Alignments",
    })
    display = display.fillna("—")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Text":            st.column_config.TextColumn(width="large"),
            "CCSS Alignments": st.column_config.TextColumn(width="medium"),
        },
    )

    fname = f"{selected_abbr}_standards.csv"
    st.download_button(
        "Download as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=fname,
        mime="text/csv",
    )