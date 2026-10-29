# Reference labels for PE Deal Intelligence

`reference_labels.csv` holds one row per labelled article, keyed by its URL; `results.md` is the latest report
(`python -m pedeals validate` rewrites it).

## How the labels were made

* **Who:** every row was labelled by Claude (an AI assistant) reading the source page's text, field by field
  (`labelled_by`). None has yet been checked by a person (`verified_by_human = no`). Before quoting the accuracy, a
  group member should re-label a random subset without looking at these labels or the program's output and report the
  agreement; disagreements are corrected here with a note, never in the program's output.
* **Development split (71 rows):** a stratified random sample (seed 42) of the deal and non-deal news in the period
  plus deliberately hard cases (the brief's example, multi-target releases, a pending sale, a combination, a rebrand,
  a podcast titled "Poised for Exit", a fund milestone, undated pages). The extraction rules were improved while
  looking at these, so their accuracy is in-sample.
* **Held-out split (38 rows):** drawn at random (seed 7) from the articles not yet labelled, *after* the rules were
  frozen in commit "PE Deal Intelligence: collector, parser, rule-based extraction and dev-set labels (rules frozen
  here)". This is the accuracy to quote. Later commits changed only the rationale-sentence choice, analysis and the
  app, none of the scored fields' rules.
* **One correction:** CAMP Digital's state was first labelled "none" from a truncated excerpt; the full page's About
  section says Edina, Minnesota, so the label was corrected to MN (noted in the row). The program had it right.

## Conventions

| Column | Convention |
|---|---|
| `is_deal`, `news_category` | Deal / Personnel / Fund / firm / Recognition / media / Other, as defined in `pedeals/taxonomy.py` |
| `published_date` | the date the firm published the page (page metadata, else the dateline, else the date beside the link on its news page; a month-only listing is the 1st of the month); `none` = no date anywhere |
| `transaction_type` | as in the taxonomy; read from what the text says happened, not from the headline's marketing ("growth investment" in a headline over a text that says "completed the acquisition" is a platform investment) |
| `buyer`, `target` | company names as written (suffixes ignored when scoring); several targets separated by `;`; `none` = not named |
| `target_industry` | the taxonomy category of the target's own business; blank = not scored (too ambiguous to label) |
| `target_state` | the state where the target is based if the text says or clearly implies it (a named city); `none` = not stated or ambiguous |
| `deal_status` | Completed only if the text says it happened (has acquired, completed, closed, recent acquisition, a headline "Completes"/"Exits"); "announced the acquisition" alone is Unknown |
| `price_disclosure` | Disclosed / Not disclosed (stated) / Not stated |

Scoring: a blank cell is not scored; `none` requires the program to leave the field empty or Unknown; names match when
equal after normalisation or when one contains the other.
