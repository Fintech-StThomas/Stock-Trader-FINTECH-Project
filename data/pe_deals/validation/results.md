# PE Deal Intelligence: validation

Reference labels: 38 held-out articles (labelled after the extraction rules were frozen in git) and 71 development articles (the rules were tuned on these, so their accuracy is in-sample).

| Field | Held-out: correct / labelled | Held-out accuracy (95% interval) | Development accuracy |
|---|---|---|---|
| is_deal | 38 / 38 | 100% (91%-100%) | 100% (71/71) |
| news_category | 38 / 38 | 100% (91%-100%) | 100% (71/71) |
| published_date | 38 / 38 | 100% (91%-100%) | 100% (71/71) |
| transaction_type | 30 / 30 | 100% (89%-100%) | 100% (54/54) |
| buyer | 30 / 30 | 100% (89%-100%) | 100% (54/54) |
| target | 30 / 30 | 100% (89%-100%) | 98% (53/54) |
| target_industry | 26 / 30 | 87% (70%-95%) | 92% (49/53) |
| target_state | 29 / 30 | 97% (83%-99%) | 96% (52/54) |
| deal_status | 29 / 30 | 97% (83%-99%) | 100% (54/54) |
| price_disclosure | 30 / 30 | 100% (89%-100%) | 100% (54/54) |

Deal vs other news (held-out): precision 100%, recall 100% (30 deals found, 0 false alarms, 0 missed).

## Consistency checks (every record, no labels needed)

| Check | Records failing |
|---|---|
| publication date after the collection date | 0 |
| publication date in the future | 0 |
| deal without a source URL | 0 |
| deal without category evidence | 0 |
| publisher recorded as an add-on buyer | 0 |
| platform investment not by the publisher | 0 |
| buyer and target the same | 0 |
| price recorded as zero | 0 |
| price disclosed without a value | 0 |
| closing date before the publication date by > 1 year | 0 |
| duplicate deal ids | 0 |
| deal with a type but no buyer and no target | 0 |

Repeatability: two builds from the same saved pages are identical (fingerprint `d5e0683fc0f5e0b2`).

## Mismatches, held-out (6)

| Article | Field | Label | Program |
|---|---|---|---|
| Align Partners with Engineering Services Firm AKS | target_state | OR |  |
| Counsel Press Expands into the Trial Courts with Case Anywhere Acquisi | target_industry | Software & data | IT services & cybersecurity |
| Marco Sealing Solutions Continues Expansion with Acquisition of Pride  | target_industry | Value-added distribution | Industrial manufacturing |
| Schneider Geospatial Expands Florida Presence | target_industry | Software & data | Unknown |
| Schneider Geospatial Expands Florida Presence | deal_status | Unknown | Completed |
| Huron Capital’s Platform 1-800 HANSONS Acquires Paramount Builders, LL | target_industry | Building & infrastructure services | Consumer & food |

## Mismatches, development (7)

| Article | Field | Label | Program |
|---|---|---|---|
| Align Capital Partners Announces Sale of CompletePet to Blue Sea Capit | target_industry | Consumer & food | Industrial manufacturing |
| Counsel Press Rebrands as Proceed, Broadens Market Reach with Three Ac | target_industry | Business services | Transportation & logistics |
| Counsel Press Rebrands as Proceed, Broadens Market Reach with Three Ac | target_state | none | TX |
| Premier Biotech Increases Offerings and Global Distribution Network wi | target | OraSure Toxicology Line; Green Earth Biomedical | OraSure Technologies, Inc |
| Goldner Hawn Forms Nexterra Environmental, A New Erosion Control and S | target_industry | Value-added distribution | Energy & environmental |
| Professional Acquires “Project Heat” | target_industry | Business services | Unknown |
| Huron Capital’s ExecFactor® Platform The Exigent Group Acquires Smith- | target_state | OH |  |

## Notes

- Every labelled URL is in the dataset.
- Labels were made by reading each source page; see data/pe_deals/validation/README.md for who labelled what and how. The intervals are 95% Wilson intervals: with 30-50 labelled deals they are wide.
