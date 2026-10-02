"""
FMCG AI Analyst  -  Streamlit + Groq LLM + DuckDB
-------------------------------------------------
User question  ->  LLM writes SQL  ->  DuckDB runs it on the CSV
              ->  Table / Chart / Summary (chosen by the checkboxes)

The CSV file must sit in the SAME folder as this app.py file.
"""

import json
import os
import re

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# =====================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>  PASTE YOUR API KEY HERE  <<<<<<<<<<<<<<<<<<<<
#
#   Get a free key at: https://console.groq.com/keys
#   Replace the text between the quotes below  (keep the quotes!)
#
GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"
#
# WARNING: if your GitHub repo is PUBLIC, do NOT commit a real key.
# Safer option: leave the line above as it is and put the key in
# Streamlit Secrets instead (see README.md). The app checks Secrets first.
# =====================================================================

CSV_FILE = "retailer_fmcg_synthetic_dashboard_data.csv"   # must be next to app.py
TABLE_NAME = "sales"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MAX_ROWS_SHOWN = 2000

st.set_page_config(page_title="FMCG AI Analyst", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    here = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(here, CSV_FILE), encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def build_schema_text(df: pd.DataFrame) -> str:
    lines = [f"Table name: {TABLE_NAME}  ({len(df):,} rows)", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        if dtype.startswith(("int", "float")):
            lines.append(f"- {col} ({dtype}): min={df[col].min()}, max={df[col].max()}")
        elif nunique <= 25:
            vals = ", ".join(map(str, sorted(df[col].dropna().unique())))
            lines.append(f"- {col} (text, {nunique} values): {vals}")
        else:
            ex = ", ".join(map(str, df[col].dropna().unique()[:5]))
            lines.append(f"- {col} (text, {nunique} values), e.g. {ex}")
    return "\n".join(lines)


def get_connection(df: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register(TABLE_NAME, df)
    return con


# ---------------------------------------------------------------- LLM
def resolve_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if os.getenv("GROQ_API_KEY"):
        return os.getenv("GROQ_API_KEY")
    if st.session_state.get("sidebar_key"):
        return st.session_state["sidebar_key"]
    return GROQ_API_KEY


def key_is_missing(key: str) -> bool:
    return (not key) or key.startswith("PASTE_YOUR")


def llm_json(client: Groq, model: str, system: str, user: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)


def sql_system_prompt(schema: str) -> str:
    return f"""You are a senior data analyst. Convert the user's question into ONE DuckDB SQL query.

{schema}

Business notes:
- Category: PC = Personal Care, HC = Home Care.
- "Sales" / "revenue" / "value" = SUM(USD_Value) unless the user asks for local currency (LC_Value) or volume (Sales_Units).
- LC_Value is in different currencies per country, so never add LC_Value across countries.
- Month is text 'YYYY-MM' (use it for trends, sorted ascending). Date is text 'YYYY-MM-DD'.
- Fiscal_Year is like 'FY2025'. The subcategory spelling 'deodrants' is intentional - use it as is.
- Use only the table "{TABLE_NAME}" and only the listed columns. Use exact category values as listed.
- Only a single SELECT (or WITH ... SELECT) statement. Never modify data.
- Give columns readable aliases, ROUND money to 2 decimals, add ORDER BY, and LIMIT 50 for rankings unless the user asks for more.
- Text comparisons: use ILIKE for flexible matching when unsure about case.

Also propose the best chart for the result.

Reply with JSON only:
{{
  "sql": "<the query>",
  "chart": {{
     "type": "bar | line | pie | scatter | area | histogram",
     "x": "<result column for x axis / pie names>",
     "y": "<result numeric column>",
     "color": "<optional result column for series, or null>",
     "title": "<short chart title>"
  }}
}}
Use "line" for time trends, "bar" for comparisons, "pie" only for <=8 shares of a total."""


FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|"
    r"pragma|install|load|call|set|truncate|read_csv|read_parquet|read_json|glob)\b",
    re.I,
)


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    if ";" in sql:
        raise ValueError("Only one statement is allowed.")
    if FORBIDDEN.search(sql):
        raise ValueError("Query contains a forbidden keyword.")
    return sql


def generate_and_run_sql(client, model, schema, question, history, con):
    """Ask LLM for SQL, run it; if it fails, show the error to the LLM and retry once."""
    system = sql_system_prompt(schema)
    context = ""
    if history:
        context = "Earlier questions in this chat (for follow-ups):\n" + "\n".join(
            f"- {h}" for h in history[-3:]) + "\n\n"
    user_msg = f"{context}Question: {question}"

    last_err = None
    plan = {}
    for attempt in range(2):
        prompt = user_msg if attempt == 0 else (
            f"{user_msg}\n\nYour previous SQL failed.\nSQL: {plan.get('sql')}\n"
            f"Error: {last_err}\nFix it and return the JSON again.")
        plan = llm_json(client, model, system, prompt)
        try:
            sql = validate_sql(plan["sql"])
            df = con.execute(sql).fetchdf()
            return sql, df, plan.get("chart") or {}
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    raise RuntimeError(f"Could not produce a working query. Last error: {last_err}")


def summarize(client, model, question, sql, df) -> str:
    sample = df.head(60).to_csv(index=False)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content":
                "You are a retail/FMCG business analyst. Answer the user's question using ONLY "
                "the data provided. Be concise (3-6 bullet points or short paragraph), quote the "
                "key numbers, call out top/bottom performers or trends, and mention USD when the "
                "values are USD_Value. Do not invent numbers."},
            {"role": "user", "content":
                f"Question: {question}\n\nSQL used:\n{sql}\n\n"
                f"Result ({len(df)} rows, first 60 shown):\n{sample}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------- charts
def build_chart(df: pd.DataFrame, spec: dict):
    if df.empty or len(df.columns) == 0:
        return None
    cols = list(df.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    ctype = str(spec.get("type", "bar")).lower()
    x = spec.get("x") if spec.get("x") in cols else None
    y = spec.get("y") if spec.get("y") in cols else None
    color = spec.get("color") if spec.get("color") in cols else None
    title = spec.get("title") or ""

    if x is None:
        x = next((c for c in cols if c not in numeric), cols[0])
    if y is None:
        y = next((c for c in numeric if c != x), None)
    if color in (x, y):
        color = None

    try:
        if ctype == "histogram":
            return px.histogram(df, x=y or x, title=title)
        if y is None:
            return None
        if ctype == "pie":
            return px.pie(df, names=x, values=y, title=title, hole=0.35)
        if ctype == "scatter":
            return px.scatter(df, x=x, y=y, color=color, title=title)
        if ctype == "line":
            return px.line(df.sort_values(x), x=x, y=y, color=color, markers=True, title=title)
        if ctype == "area":
            return px.area(df.sort_values(x), x=x, y=y, color=color, title=title)
        return px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
    except Exception:
        return None


# ---------------------------------------------------------------- UI helpers
def render_assistant(item: dict, key_prefix: str):
    if item.get("error"):
        st.error(item["error"])
        return
    df = item["df"]
    if item.get("show_sql"):
        with st.expander("SQL generated by the LLM"):
            st.code(item["sql"], language="sql")
    if df.empty:
        st.info("The query ran fine but returned no rows.")
        return
    if item.get("summary"):
        st.markdown("#### 📝 Summary")
        st.markdown(item["summary"])
    if item.get("fig") is not None:
        st.markdown("#### 📈 Chart")
        st.plotly_chart(item["fig"], use_container_width=True, key=f"{key_prefix}_chart")
    if item.get("want_table"):
        st.markdown("#### 📋 Table")
        st.dataframe(df.head(MAX_ROWS_SHOWN), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="result.csv", mime="text/csv", key=f"{key_prefix}_dl")


# ---------------------------------------------------------------- main app
def main():
    df = load_data()
    schema = build_schema_text(df)

    st.title("📊 FMCG AI Analyst")
    st.caption("Ask a question in plain English. The LLM writes SQL, runs it on the CSV, "
               "and shows a table, chart and/or summary.")

    # ---- sidebar
    with st.sidebar:
        st.header("Output options")
        want_table = st.checkbox("📋 Table", value=True)
        want_chart = st.checkbox("📈 Visual (chart)", value=True)
        want_summary = st.checkbox("📝 Summary / answer", value=True)
        show_sql = st.checkbox("Show generated SQL", value=False)

        st.divider()
        model = st.selectbox("Groq model", MODELS)

        if key_is_missing(resolve_api_key()):
            st.warning("No API key found. Paste it in app.py (marked section) or here:")
            st.text_input("Groq API key", type="password", key="sidebar_key")

        st.divider()
        st.subheader("Try these")
        examples = [
            "Top 10 brands by USD sales",
            "Monthly sales trend by category",
            "Sales by country and retailer",
            "Market share of manufacturers in toothpaste",
            "Which city sells the most Dove bar soap units?",
            "Compare FY2025 vs FY2026 sales by subcategory",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["queued"] = ex

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        with st.expander("Dataset info"):
            st.write(f"{len(df):,} rows × {len(df.columns)} columns")
            st.write(f"Months: {df['Month'].min()} → {df['Month'].max()}")
            st.dataframe(df.head(5), hide_index=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ---- replay history
    for i, m in enumerate(st.session_state["messages"]):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["content"])
            else:
                render_assistant(m, key_prefix=f"h{i}")

    # ---- new question
    question = st.chat_input("e.g. Show monthly sales of Colgate in Saudi Arabia")
    question = question or st.session_state.pop("queued", None)
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        api_key = resolve_api_key()
        if key_is_missing(api_key):
            item = {"role": "assistant", "error": "Add your Groq API key (see the marked section in app.py)."}
        elif not (want_table or want_chart or want_summary):
            item = {"role": "assistant", "error": "Tick at least one option: Table, Visual or Summary."}
        else:
            try:
                client = Groq(api_key=api_key)
                con = get_connection(df)
                history = [m["content"] for m in st.session_state["messages"][:-1]
                           if m["role"] == "user"]
                with st.spinner("Writing SQL and querying the data..."):
                    sql, result, chart_spec = generate_and_run_sql(
                        client, model, schema, question, history, con)

                item = {"role": "assistant", "sql": sql, "df": result, "show_sql": show_sql,
                        "want_table": want_table, "fig": None, "summary": None}

                if not result.empty:
                    if want_chart:
                        item["fig"] = build_chart(result, chart_spec)
                    if want_summary:
                        with st.spinner("Writing summary..."):
                            item["summary"] = summarize(client, model, question, sql, result)
            except Exception as e:  # noqa: BLE001
                item = {"role": "assistant", "error": f"Something went wrong: {e}"}

        render_assistant(item, key_prefix=f"n{len(st.session_state['messages'])}")
    st.session_state["messages"].append(item)


main()
