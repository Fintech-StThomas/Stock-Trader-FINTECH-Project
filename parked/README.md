# Parked projects

Two private-equity projects, parked in November 2026 so that the deployed app is Lighthouse (the stock app) only. They
are complete and runnable, but nothing outside this folder uses them:

* the Lighthouse app (`app.py`, `lighthouse/`) imports nothing from here, and nothing here imports Lighthouse (each
  project has a test that fails if that ever changes);
* the repository's test suite and the CI workflow on every push do not collect these folders
  (`norecursedirs` in the root `pyproject.toml`); their own tests run only when started by hand (below);
* the folder names contain a hyphen, so they cannot be imported as Python packages by accident;
* Streamlit Community Cloud runs the root `app.py`, so nothing here is deployed.

| Project | What it is | Run it | Tests |
|---|---|---|---|
| [`pe-deal-sourcing/`](pe-deal-sourcing/) | **PE Deal Sourcing**: sixteen sponsors' deal announcements (2015-2026) turned into a sourcing record: consolidation map, exit watch from a rebuilt portfolio ledger, competing buyers, sourcing brief, three rounds of held-out validation. Runs offline on saved pages; no API key; free. Method: [`docs/PE_DEALS.md`](pe-deal-sourcing/docs/PE_DEALS.md). | `cd parked/pe-deal-sourcing && pip install -r requirements.txt && streamlit run app.py` | 32 |
| [`pe-deal-explorer/`](pe-deal-explorer/) | **PE Deal Intelligence**: the PE workspace that was in the app (AI Deal Research and Deal Detail over a saved snapshot of deals, plus the collection, extraction, validation and publishing modules). Its AI search calls an OpenAI model, **a paid API**, and only works when a key is configured; the saved deals and Deal Detail work without one. Docs: [`docs/`](pe-deal-explorer/docs/). | `cd parked/pe-deal-explorer && pip install -r requirements.txt && streamlit run app.py` | 88 |

Run a project's tests from its folder: `cd parked/<project> && python -m pytest`. On GitHub: **Actions → Parked PE
projects → Run workflow** (manual only; it never runs on a push).

Notebook (PE Deal Sourcing): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fintech-StThomas/Stock-Trader-FINTECH-Project/blob/main/parked/pe-deal-sourcing/notebooks/PE_Deal_Intelligence.ipynb)

## Bringing one back later

Each project is a normal folder: to reconnect one, move its package and data back to the repository root (or add the
folder to the app's import path), wire a page or tab in `app.py`, and move its tests into `tests/`. The explorer keeps
`workspace_ui.py`, the Stock / PE switch it used in the app. The sourcing project's `pedeals.app_view.render()` draws
its whole page. Before reconnecting the explorer, decide whether a paid AI search fits the project; the sourcing
project needs no paid service.
