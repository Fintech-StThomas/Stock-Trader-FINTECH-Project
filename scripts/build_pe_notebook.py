"""Builds notebooks/PE_Deal_Intelligence.ipynb (kept as a script so the notebook is reproducible and diff-friendly).

The committed notebook is this output executed once on the saved pages, so it opens with its results:
    python scripts/build_pe_notebook.py
    cd notebooks && LIGHTHOUSE_OFFLINE=1 python -m nbconvert --to notebook --execute --inplace PE_Deal_Intelligence.ipynb
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))     # noqa: E731

md("""# 🎯 PE Deal Sourcing

*Public private-equity deal announcements, turned into an evidence-backed sourcing record for a deal team.*

**User.** A deal team at a lower-middle-market sponsor sourcing in five sectors: building & facility services, IT
services & cybersecurity, tech-enabled business services, industrial manufacturing and value-added distribution.
**Recurring task.** Every week: which niches are peer sponsors rolling up, which sponsor-owned companies have been held
long enough to be sold soon, and who would we bid against for a target?

This notebook runs the whole workflow:

1. **Setup** — fresh Colab session: clone the project, install dependencies
2. **Collect** — sixteen buy-and-build sponsors' news pages (polite: robots.txt and crawl delays), or the saved pages (default)
3. **Process** — parse, tell deals from other news, extract every field with its evidence, tag sector and niche
4. **Inspect** — the data dictionary, a record next to its source sentences, coverage and failed pages
5. **Validate** — accuracy against hand-labelled pages (held-out vs development), consistency checks, repeatability
6. **Source** — the consolidation map, the exit watch (portfolio ledger), competing buyers, the sourcing brief
7. **Extras** — similar-deal search, the new-deal digest, exports

> **Cached data.** By default everything runs on pages saved by the collector (`data/pe_deals/cache/`, collection time
> in every record), so the notebook gives the same result even if a website changes or goes offline. Set `LIVE = True`
> in step 2 to fetch new pages first (slow by design: up to 10 seconds between pages on some sites).

The same product is the **🤝 PE deals** tab of the web app (`streamlit run app.py`). Method, data dictionary and
limitations: `docs/PE_DEALS.md`.
""")
md("## 1 · Setup")
code("""# On Google Colab this clones the project; inside the repo (or its notebooks/ folder) it just finds the package.
import os, sys, subprocess, pathlib
REPO_URL = "https://github.com/lotainc0/Stock-Trader-FINTECH-Project.git"
here = pathlib.Path.cwd()
root = next((p for p in [here, *here.parents] if (p / "pedeals" / "__init__.py").exists()), None)
if root is None:
    subprocess.run(["git", "clone", "-q", REPO_URL, "lighthouse_repo"], check=True)
    root = here / "lighthouse_repo"
os.chdir(root); sys.path.insert(0, str(root))
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy", "requests", "beautifulsoup4", "lxml",
                    "plotly"], check=True, timeout=900)
except Exception as exc:  # offline or already installed - keep going
    print("pip install skipped:", exc)
import pandas as pd
from IPython.display import Markdown, display
import pedeals
from pedeals import build_dataset, analytics as an, charts
from pedeals.sources import SOURCES
pd.set_option("display.width", 180); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)
print("PE Deal Intelligence", pedeals.__version__, "ready in", root)""")
md("""## 2 · Collect

The four firms are peers of the user (lower-middle-market, Midwest / central US, explicit buy-and-build) and each lists
its news in an XML sitemap. The collector reads each site's robots.txt and obeys it, waits the site's crawl delay
between requests, names itself in the User-Agent, saves every page with its fetch time and logs every URL's outcome.""")
code("""pd.DataFrame([{"key": k, "firm": s.firm, "home office": s.hq, "sitemaps": ", ".join(s.sitemaps)} for k, s in SOURCES.items()])""")
code("""LIVE = False           # True: fetch new articles from the live sites first (minutes; pages already saved are skipped)
START = "2015-01-01"   # first publication date collected (the ledger needs the older years)

if LIVE:
    from pedeals.collect import collect
    discovery = collect(start=START)
    display(discovery.groupby(["source", "outcome"]).size().rename("urls").to_frame())
else:
    print("Using the saved pages (CACHED DATA). Set LIVE = True to refresh.")""")
md("""## 3 · Process

`build_dataset` parses every saved page, finds its publication date (page metadata → dateline → the firm's news-page
listing → first date in the text; never guessed), classifies it (deal, personnel, fund / firm, recognition, other),
extracts the deal fields from the release's lead sentence with the evidence for each, infers ownership from the
firm's other announcements where the release is silent, marks duplicates and keeps the period.""")
code("""ds = build_dataset(start=START)
print(ds.summary_text())""")
code("""# Coverage: what each sitemap listed, what was skipped and why, what was kept
ds.coverage""")
code("""# Pages that could not be fetched or parsed (with the reason), and articles with no publication date
display(ds.failures)
ds.records.loc[ds.records["published_date"] == "", ["publisher", "headline", "period_note"]]""")
md("""## 4 · Inspect

### Data dictionary
Every output field is defined in `docs/PE_DEALS.md` §5. The categories are defined in code, in one place:""")
code("""from pedeals.taxonomy import NEWS_CATEGORIES, TRANSACTION_TYPES, DEAL_STATUS, INDUSTRIES, RATIONALE_TAGS
display(pd.DataFrame(TRANSACTION_TYPES.items(), columns=["transaction_type", "definition"]))
display(pd.DataFrame(DEAL_STATUS.items(), columns=["deal_status", "definition"]))
display(pd.DataFrame(NEWS_CATEGORIES.items(), columns=["news_category", "definition"]))
pd.DataFrame([(k, ", ".join(v[:8]) + " ...") for k, v in INDUSTRIES.items()], columns=["target_industry", "example stems"])""")
md("""### The brief's example, as a record

Tonka Bay published the Standard Locknut / Jones Bearing announcement; Tonka Bay is **not** the buyer. Each value
below sits next to the words it was read from.""")
code("""deals = ds.unique_deals
jones = deals[deals["headline"].str.contains("Jones Bearing")].iloc[0]
Markdown(an.deal_card(jones))""")
code("""# All deal announcements in the period (the main columns)
deals[an.DISPLAY_COLUMNS].head(15)""")
md("""## 5 · Validate

Articles were labelled field by field from the source pages. The rules were improved on a **development** set
(in-sample) and frozen in git before each **held-out** set was drawn and labelled; only held-out accuracy is the tool's
accuracy. Consistency checks run on every record; repeatability compares two builds.""")
code("""from pedeals.validate import full_report
Markdown(full_report(ds))""")
md("""## 6 · Source

### Where is the consolidation? (add-ons by niche of the five focus sectors)
Momentum compares the last twelve months of add-ons with the twelve before.""")
code("""from pedeals import sourcing as so
since_2022 = deals[deals["published_date"] >= "2022-01-01"]
cons = so.consolidation(since_2022)
display(cons.head(15))
charts.horizontal_counts(cons.head(15), "niche", "add_ons", "Add-ons by niche since 2022", "add-ons").show()""")
md("""### Exit watch: sponsor-owned companies held long enough to be sold soon

The portfolio ledger rebuilds each sponsor's holdings from its own announcements: an investment announcement or a
sentence that dates it ("ACP acquired StenTech in May of 2022") opens a holding, add-ons show it is still owned, a
sale closes it, a rename ("rebranded as CompletePet") links the names. Lower-middle-market sponsors usually sell after
four to six years.""")
code("""led = so.ledger(ds.records)
exited = led[(led["status"] == "Exited") & (led["years_basis"] == "from entry")]["years_held"]
print(f"{len(led)} holdings found; {len(exited)} exits with a dated entry, median hold {exited.median():.1f} years")
watch = so.exit_watch(led, min_years=4, sectors=so.FOCUS_SECTORS)
display(watch[["company", "sponsor", "focus_sector", "niche", "years_held", "entry_date", "entry_basis", "add_ons", "signal"]])
charts.years_held(watch).show()""")
md("""### Competing buyers and the sourcing brief

*"We want an MSP. Who else is buying them, what did they buy, and which sponsor-owned MSPs may come to market?"*
Change `SECTOR` and `NICHE`.""")
code("""SECTOR, NICHE = "IT services & cybersecurity", "Managed IT services (MSP)"
display(so.competing_buyers(since_2022, led, [NICHE], [SECTOR]))
Markdown(so.sourcing_brief(since_2022, led, SECTOR, NICHE))""")
code("""# What the announcements leave out: the honest size of the gaps (a missing price is never zero)
an.disclosure_gaps(since_2022[since_2022["focus_sector"].isin(so.FOCUS_SECTORS)])""")
md("""**What the evidence supports:** who announced which acquisitions, where and when, and how long each sponsor says it
has owned its platforms. **What needs further research:** companies nobody has bought yet (a company database), whether
"no exit announced" companies are really still held, prices, and the sponsors' fund life. The collected announcements
are not a measure of the PE market.""")
md("## 7 · Extras: similar deals, the new-deal digest, exports")
code("""# More targets like Jones Bearing (target description and rationale in similar words)
an.similar_deals(deals, jones["deal_id"], k=5)[["similarity", "published_date", "publisher", "buyer", "target", "niche", "target_state"]]""")
code("""# Monitoring: everything published in the last 120 days of the data
latest = pd.to_datetime(deals["published_date"]).max()
an.new_since(deals, (latest - pd.Timedelta(days=120)).date().isoformat())[["published_date", "publisher", "buyer", "target", "transaction_type", "niche"]]""")
code("""# Export: CSV / JSON / Markdown brief (saved sample outputs live in data/pe_deals/output/)
for p in ds.export("pe_deals_output"):
    print(p)""")
md("""## Limitations

* Sixteen sponsors' self-published announcements: selective and incomplete, not the PE market.
* "No exit announced" is not proof a company is still held; entry dates found in passing are the sponsor's statement.
* Prices, closing dates and stakes are almost never disclosed; the fields show the gap instead of filling it.
* Pages with no publication date are excluded from the period and listed, never guessed.
* The rules fit these sites' press-release formulas; a new firm needs a source entry and a labelled check first.
* The reference labels were produced with AI assistance and still need a human spot-check (`data/pe_deals/validation/README.md`).
""")
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"      # stable ids: rebuilding an unchanged notebook gives an identical file
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "colab": {"name": "PE_Deal_Intelligence.ipynb", "toc_visible": True}}
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/PE_Deal_Intelligence.ipynb")
print("wrote notebooks/PE_Deal_Intelligence.ipynb with", len(cells), "cells")
