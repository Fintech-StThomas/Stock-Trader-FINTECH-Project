# PE Deal Intelligence: executive summary (draft for the group to edit)

**The user and the problem.** A portfolio operations manager at a lower-middle-market sponsor plans its platforms'
next add-ons. The best public evidence of what peers are doing is their own deal announcements, but reading them means
visiting several websites, separating deals from news about people and funds, and keeping notes with no links back to
the source. PE Deal Intelligence does that reading and keeps the evidence.

**The product.** A tab of the existing Lighthouse dashboard (and a Colab notebook) that collects the news pages of four
peer sponsors (Tonka Bay Equity Partners, Goldner Hawn, Huron Capital, Align Capital Partners), keeps the deal
announcements published since January 2023, and turns each into a record: publisher, buyer, the buyer's relationship
to the publisher, target, transaction type, the target's business and industry (separate from its customers'
industries), location, the stated rationale tagged by capability, status, and the disclosure gaps (price, closing date,
stake). Every field sits next to the sentence it came from and a link to the page. Users search and filter the
deals, follow each platform's buy-and-build cadence, generate an add-on research brief for their own industry, and see
what is new since a date.

**Scope and method.** 518 sitemap URLs, 185 pages fetched politely (robots.txt and 10-second crawl delays obeyed),
148 news items in the period, 115 deal announcements: 80 add-ons by 28 platforms, 17 platform investments, 14 exits,
2 growth investments, 2 combinations. Extraction uses transparent rules that read the press-release lead sentence
("[sponsor's] platform X has acquired Y, a provider of ..."), not a black-box model, so every value can be traced and
the run is free and repeatable. Pages are saved, so the tool runs offline on cached data and says so.

**Key findings (from the announcements, not the market).** Add-on buying dominates (80 of 115). The stated reasons
are technology or capability (56% of add-ons that state a reason), product or service breadth (56%), customers and
channels (51%) and geographic reach (47%). The fastest builders announce an add-on every two to four months
(Goldner Hawn's Cyber Advisors: eight IT-services add-ons, a median of 79 days apart). No announcement disclosed a
price; four said terms were not disclosed.

**Validation.** 109 articles were hand-labelled from the source pages; the rules were tuned on 71 and then frozen
before 38 more were labelled. On those held-out articles, deal detection, dates, transaction type, buyer and target
were all correct (30 of 30 deals; 95% interval 89-100%), target industry 87% and location 97%. Consistency checks find
no impossible records and two runs give identical output. The labels were made with AI assistance and still need a
human spot-check, and most held-out items come from one firm.

**Business value.** A recurring research task that takes hours becomes a few minutes, with citations an investment
committee can check. The brief turns peers' add-on activity into a shortlist of capabilities and regions to
investigate, which is where operational alpha ideas start.

**Limitations.** Four firms' self-selected announcements are not the PE market; a stated rationale is a claim, not
evidence of better margins or returns, which needs operating data; prices and closing dates are almost never public;
14 pages carry no date and are excluded rather than guessed; a new firm needs its pages checked against labels
before its records are trusted.

**Creative features.** Buy-and-build tracker across platform renames, add-on research brief, evidence cards with
similar-deal search, new-deal digest and live refresh, ownership inferred across announcements, disclosure-gap report.
