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

md("""# 🤝 PE Deal Intelligence

*Public private-equity deal announcements, turned into an evidence-backed deal record for a portfolio operations team.*

**User.** A portfolio operations manager at a lower-middle-market sponsor that buys and builds industrial, distribution and
business-services companies. **Recurring task.** Before each add-on planning cycle: which add-ons are peer sponsors'
platforms buying, what capabilities do they say those add-ons bring, and how fast do the best builders close deals?

This notebook runs the whole workflow:

1. **Setup** — fresh Colab session: clone the project, install dependencies
2. **Collect** — the four firms' news pages (polite: robots.txt and crawl delays), or the saved pages (default)
3. **Process** — parse, tell deals from other news, extract every field with its evidence, handle duplicates and gaps
4. **Inspect** — the data dictionary, a record next to its source sentences, coverage and failed pages
5. **Validate** — accuracy against hand-labelled pages (held-out vs development), consistency checks, repeatability
6. **Decide** — the add-on research brief, the buy-and-build tracker, charts
7. **Extras** — similar-deal search, the new-deal digest, exports

> **Cached data.** By default everything runs on pages saved by the collector (`data/pe_deals/cache/`, collection time
> in every record), so the notebook gives the same result even if a website changes or goes offline. Set `LIVE = True`
> in step 2 to fetch new pages first (slow by design: about 10 seconds between pages on two of the sites).

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
START = "2023-01-01"   # first publication date of interest

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

109 articles were labelled field by field from the source pages. The rules were improved on 71 (**development**,
in-sample) and then frozen in git before 38 more were drawn and labelled (**held-out**). Only the held-out accuracy is
the tool's accuracy. Consistency checks run on every record; repeatability compares two builds.""")
code("""from pedeals.validate import full_report
Markdown(full_report(ds))""")
md("""## 6 · Decide: the add-on research brief

*"Our platform is an IT managed-services provider. Which capabilities are peer platforms adding, and where should our
next add-on search start?"* Change `INDUSTRY` to your platform's industry.""")
code("""INDUSTRY = "IT services & cybersecurity"
Markdown(an.research_brief(deals, INDUSTRY, "Which capabilities are peer platforms adding, and which should we investigate?"))""")
code("""# Buy-and-build tracker: add-ons per platform and how fast they come
bb = an.buy_and_build(deals)
bb.drop(columns=["platform_key"]).head(12)""")
code("""adds = deals[deals["transaction_type"] == "Add-on acquisition"].merge(bb[["platform_key", "platform"]], on="platform_key")
charts.buy_and_build_timeline(adds, bb[bb["add_ons"] >= 2]["platform"].tolist()).show()""")
code("""charts.deals_by_quarter(deals).show()
rf = an.rationale_frequency(deals[deals["transaction_type"] == "Add-on acquisition"])
charts.horizontal_counts(rf, "tag", "deals", "Why add-ons were made (stated rationale)", "add-ons").show()
display(an.rationale_frequency(deals[deals["transaction_type"] == "Add-on acquisition"], by="target_industry"))""")
code("""# What the announcements leave out: the honest size of the gaps (a missing price is never zero)
an.disclosure_gaps(deals)""")
md("""**What the evidence supports:** which companies were bought, by whom, when, and the capabilities the sponsors say they
add. **What needs further research:** prices (not disclosed), whether add-ons improved margins, growth or returns
(operating data), deals the firms did not announce. The collected announcements are not a measure of the PE market.""")
md("## 7 · Extras: similar deals, the new-deal digest, exports")
code("""# Who else bought something like Jones Bearing? (target description and rationale in similar words)
an.similar_deals(deals, jones["deal_id"], k=5)[["similarity", "published_date", "publisher", "buyer", "target", "target_industry"]]""")
code("""# Monitoring: everything published in the last 120 days of the data
latest = pd.to_datetime(deals["published_date"]).max()
an.new_since(deals, (latest - pd.Timedelta(days=120)).date().isoformat())[["published_date", "publisher", "buyer", "target", "transaction_type"]]""")
code("""# Export: CSV / JSON / Markdown brief (saved sample outputs live in data/pe_deals/output/)
for p in ds.export("pe_deals_output"):
    print(p)""")
md("""## Limitations

* Four firms' self-published announcements: selective and incomplete, not the PE market.
* A stated rationale is the sponsor's claim, not evidence that a deal improved results.
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
