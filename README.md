# FMCG AI Analyst (Streamlit + Groq)

Ask questions in plain English about the FMCG retail CSV. The LLM writes SQL,
DuckDB runs it on the CSV, and you get a **table**, **chart** and/or **summary**.

## Folder layout (everything in the same GitHub folder)
```
app.py
retailer_fmcg_synthetic_dashboard_data.csv   <- the "database"
requirements.txt
.streamlit/secrets.toml.example
```

## 1. Add your Groq API key
Get a key: https://console.groq.com/keys

**Option A (easy):** open `app.py`, find the box marked `PASTE YOUR API KEY HERE`
and replace `PASTE_YOUR_GROQ_API_KEY_HERE` with your key.
> Don't do this in a public GitHub repo, anyone could copy your key.

**Option B (safe, recommended):** leave `app.py` untouched and
- locally: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put your key in it
- Streamlit Cloud: App settings -> Secrets -> add `GROQ_API_KEY = "gsk_..."`

## 2. Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy free on Streamlit Community Cloud
1. Push this folder to GitHub.
2. Go to https://share.streamlit.io -> New app -> pick the repo, branch, main file `app.py`.
3. Add the key in Secrets (Option B) -> Deploy.

## Example questions
- Top 10 brands by USD sales
- Monthly sales trend by category
- Market share of manufacturers in toothpaste
- Compare FY2025 vs FY2026 sales by subcategory

## Notes
- Only SELECT queries are allowed; anything else is blocked.
- LC_Value is in local currencies, so cross-country totals use USD_Value.
- If you swap the CSV, keep the same file name or change `CSV_FILE` in `app.py`.
