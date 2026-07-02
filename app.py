"""
EM2 Standards Alignment Explorer
Streamlit app for exploring Learnosity alignment data.

Tabs:
    1. CCSS → State Standards  — pick a CCSS standard, see all aligned state standards + map
    2. State Standards Browser  — pick a state, browse all its standards with CCSS matches
    3. Query                   — SQL or plain-language queries powered by Claude
"""

import os
import sqlite3

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── CONFIG ────────────────────────────────────────────────────────────────────

DB_PATH = "learnosity.db"

ALL_US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
    "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]

# ── SCHEMA PROMPT ─────────────────────────────────────────────────────────────
# System prompt used when translating plain-language questions to SQL.

SCHEMA_PROMPT = """\
You are a SQL assistant for a SQLite database called learnosity.db. It stores \
Eureka Math 2 (EM2) standards alignment data — crosswalk mappings between state \
math standards and Common Core State Standards (CCSS).

Generate a syntactically correct SQLite SELECT query that answers the user's \
question. Return ONLY the SQL query — no explanation, no markdown fences, \
no commentary of any kind.

Schema:

states  (42 rows — one per state / standards-year combination)
  state_id         INTEGER  PRIMARY KEY
  abbreviation     TEXT     Two-letter state code (AK, AL, AR, ...)
  standards_year   REAL     Year of standards version; 0 = no year annotation.
                            AR, IN, ND, WV each have two rows (old + new standards).

ccss_standards  (676 rows — includes leaf standards, clusters, and domains)
  ccss_id      INTEGER  PRIMARY KEY
  parent_id    INTEGER  REFERENCES ccss_standards — NULL for domains; domain id for clusters;
                        standard id for child standards (e.g. 1.NBT.B.2.a)
  code         TEXT     e.g. '1.OA.A.1', '3.MD.C', '8.NS'.
                        HS prefix is stripped: 'A.APR.A.1' not 'HSA.APR.A.1'.
  grade        TEXT     'K', '1'-'8', or 'HSA'/'HSF'/'HSN'/'HSS'/'HSG' for high school
  domain       TEXT     Domain abbreviation, e.g. 'OA', 'NBT', 'MD', 'APR'
  domain_name  TEXT     Full domain name, e.g. 'Operations and Algebraic Thinking'
  cluster      TEXT     Cluster letter ('A', 'B', 'C'...); NULL for domain rows
  standard     INTEGER  Standard number; NULL for domain and cluster rows
  is_subpart   INTEGER  1 if child standard (e.g. '1.NBT.B.2.a'), else 0
  text         TEXT     Full text of the standard, cluster, or domain

state_standards  (19,821 rows — one per unique state standard code)
  state_standard_id  INTEGER  PRIMARY KEY
  state_id           INTEGER  REFERENCES states
  agency_code        TEXT
  level_name         TEXT     Hierarchy level name in the state's framework
  standard_code      TEXT     Learnosity internal tag, e.g. 'AK.CS.MA.9-12.A-APR'
  grade              TEXT     May be NULL for some states
  standard_text      TEXT     Full text; NULL for some states (coverage ~30%)

crosswalk  (22,780 rows — maps CCSS standards to state standards)
  crosswalk_id       INTEGER  PRIMARY KEY
  ccss_id            INTEGER  REFERENCES ccss_standards
  state_standard_id  INTEGER  REFERENCES state_standards
  source_file        TEXT     Source CSV filename

SQLite notes:
- Use GROUP_CONCAT() not STRING_AGG()
- Use lower() or LIKE for case-insensitive search
- No FULL OUTER JOIN — use LEFT JOIN + UNION ALL if needed
- standard and cluster can be NULL; use IS NULL / IS NOT NULL accordingly
"""


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
            ORDER BY ss.grade, ss.standard_code
            """,
            conn,
            params=(state_id,),
        )


# ── QUERY TAB HELPERS ─────────────────────────────────────────────────────────

def get_api_key() -> str | None:
    """
    Looks for the Anthropic API key in Streamlit secrets first (for Cloud
    deployment), then falls back to the ANTHROPIC_API_KEY environment variable
    (for local development).
    """
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource
def get_anthropic_client() -> anthropic.Anthropic | None:
    """Cached Anthropic client — returns None if no API key is configured."""
    key = get_api_key()
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def generate_sql(question: str) -> str:
    """Send a plain-language question to Claude and return the SQL it generates."""
    client = get_anthropic_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SCHEMA_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text.strip()


def is_read_only(sql: str) -> bool:
    """
    Return True for SELECT queries and WITH (CTE) queries.
    WITH always precedes a SELECT in practice; blocking it would prevent
    Claude from using subqueries, which it needs for anything non-trivial.
    """
    first_word = sql.strip().split()[0].lower()
    return first_word in ("select", "with")


def run_query(sql: str) -> tuple[pd.DataFrame, str | None]:
    """
    Execute a SQL query and return (DataFrame, error_message).
    DataFrame is empty on error; error_message is None on success.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql(sql, conn), None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────

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

tab1, tab2, tab3 = st.tabs([
    "CCSS → State Standards",
    "State Standards Browser",
    "Query",
])


# ── TAB 1: CCSS → STATE STANDARDS ────────────────────────────────────────────

with tab1:
    ccss_df = load_ccss_standards()

    records    = ccss_df.to_dict("records")
    labels     = [build_ccss_label(r) for r in records]
    id_by_lbl  = dict(zip(labels, ccss_df["ccss_id"]))
    row_by_lbl = dict(zip(labels, records))

    chosen_label = st.selectbox(
        "Search for a CCSS standard",
        options=[""] + labels,
        format_func=lambda x: x or "— type a grade, code, or keyword to search —",
    )

    if chosen_label:
        ccss_id = id_by_lbl[chosen_label]
        std = row_by_lbl[chosen_label]

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
    states_df = load_states()
    abbrs     = sorted(states_df["abbreviation"].dropna().unique().tolist())

    selected_abbr = st.selectbox("Select a state", abbrs)

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

    st.download_button(
        "Download as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_abbr}_standards.csv",
        mime="text/csv",
    )


# ── TAB 3: QUERY ─────────────────────────────────────────────────────────────

with tab3:
    api_client = get_anthropic_client()

    mode = st.radio(
        "Mode",
        ["Plain language", "SQL"],
        horizontal=True,
    )

    if mode == "Plain language":
        if api_client is None:
            st.warning(
                "No Anthropic API key found. Add `ANTHROPIC_API_KEY` to your "
                "Streamlit secrets (Settings → Secrets on Streamlit Cloud) or "
                "set it as an environment variable for local use."
            )
        else:
            question = st.text_area(
                "Ask a question about the data",
                placeholder=(
                    "e.g. Which states have alignments for the most 3rd grade standards?\n"
                    "e.g. How many crosswalk entries does each CCSS domain have?\n"
                    "e.g. Which CCSS standards have no state alignments at all?"
                ),
                height=120,
            )

            if st.button("Generate and run", type="primary"):
                if not question.strip():
                    st.warning("Enter a question first.")
                else:
                    with st.spinner("Generating SQL…"):
                        try:
                            sql = generate_sql(question)
                        except Exception as e:
                            st.error(f"Claude API error: {e}")
                            st.stop()

                    with st.expander("Generated SQL", expanded=True):
                        st.code(sql, language="sql")

                    if not is_read_only(sql):
                        st.error(
                            "Claude generated a non-SELECT statement. "
                            "Refusing to run it for safety."
                        )
                    else:
                        df, err = run_query(sql)
                        if err:
                            st.error(f"Query error: {err}")
                        elif df.empty:
                            st.info("Query returned no rows.")
                        else:
                            st.caption(f"{len(df)} row(s) returned")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download as CSV",
                                data=df.to_csv(index=False).encode("utf-8"),
                                file_name="query_results.csv",
                                mime="text/csv",
                            )

    else:  # SQL mode
        sql_input = st.text_area(
            "SQL",
            placeholder="SELECT ...",
            height=150,
        )

        if st.button("Run", type="primary"):
            if not sql_input.strip():
                st.warning("Enter a query first.")
            elif not is_read_only(sql_input):
                st.error("Only SELECT queries are permitted.")
            else:
                df, err = run_query(sql_input)
                if err:
                    st.error(f"Query error: {err}")
                elif df.empty:
                    st.info("Query returned no rows.")
                else:
                    st.caption(f"{len(df)} row(s) returned")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download as CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="query_results.csv",
                        mime="text/csv",
                    )