# Reference labels for PE Deal Sourcing

`reference_labels.csv` holds one row per labelled article, keyed by its URL; `results.md` is the latest report
(`python -m pedeals validate` rewrites it).

## How the labels were made

* **Who:** rows were labelled by AI models reading the source page's text, field by field (`labelled_by`). Round 1
  and the development set: Claude, the same model that wrote the rules. Rounds 2 and 3: Claude Sonnet, a separate
  model run as two independent labellers, each given only the article text (title, body, About sections, the page's own
  date metadata) and the written guide, never the program's output or code. None has yet been checked by a person
  (`verified_by_human = no`). Before quoting the accuracy, a group member should re-label a random subset without
  looking at these labels or the program's output and report the agreement; disagreements are corrected here with a
  note, never in the program's output.
* **Development split (71 rows):** a stratified random sample (seed 42) of the deal and non-deal news in the period
  plus deliberately hard cases (the brief's example, multi-target releases, a pending sale, a combination, a rebrand,
  a podcast titled "Poised for Exit", a fund milestone, undated pages). The extraction rules were improved while
  looking at these, so their accuracy is in-sample.
* **Held-out round 1 (`held-out`, 38 rows):** drawn at random (seed 7) from the first four sponsors' unlabelled
  articles after the rules were frozen in commit "PE Deal Intelligence: collector, parser, rule-based extraction and
  dev-set labels (rules frozen here)".
* **Held-out round 2 (`held-out-2`, 60 rows):** after the move to deal sourcing (twelve more sponsors, collection back
  to 2015, focus sectors and niches), the rules were frozen in commit `0bf8a14`; 48 articles the program calls deals and
  12 it does not were then drawn at random (seed 2026) from the new sponsors and the 2015-2022 pages. Scored at the
  freeze first (kept in `pedeals/validate.py` as `FROZEN_SCORES`): industry 26/47, focus sector 30/47 and niche 13/31
  were weak. The industry rules were then fixed after reading those mismatches, so round 2's current score is in-sample.
* **Held-out round 3 (`held-out-3`, 44 rows):** the fixed rules were frozen in commit `1835b7f`; 36 articles the program
  calls deals and 8 it does not were then drawn at random (seed 3033) from every unlabelled dated article since 2015.
  **This is the accuracy to quote.** The guide given to its labellers names two niches added with the fixes (Facility
  services; Telecom & utility infrastructure) and says insurance agencies are Financial services, as both round-2
  labellers had already decided; nothing else changed.
* **Known labelling disagreement:** the guide says "pleased to announce the sale of" alone is status Unknown; round-2
  labellers marked several such sales Completed. The program follows the guide; those rows count as program errors.
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
| `focus_sector`, `niche` | rounds 2-3 only: the focus sector the target's industry maps to (or Outside focus) and its niche (`pedeals/taxonomy.py`); blank niche outside the focus sectors |

Scoring: a blank cell is not scored; `none` requires the program to leave the field empty or Unknown; names match when
equal after normalisation ("&" = "and") or when one contains the other, also against the short name the release
itself defines for the company and a sponsor's other names (HCI Equity Partners is now Oridian Capital Partners).
