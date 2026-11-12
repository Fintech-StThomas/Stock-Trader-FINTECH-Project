# PE Deal Sourcing: executive summary (draft for the group to edit)

**The user and the problem.** A deal team at a lower-middle-market sponsor sources new platforms and add-ons in five
sectors: building & facility services, IT services & cybersecurity, tech-enabled business services, industrial
manufacturing and value-added distribution. Each week it asks which niches other sponsors are rolling up, which
sponsor-owned companies have been held long enough to be sold, and who it would bid against. The evidence is in
sponsors' own press releases, spread across dozens of websites, with no common record and no links back to the source.

**The product.** A standalone Streamlit app, parked separately from the Lighthouse stock app (and a Colab notebook), that reads the news pages of sixteen
buy-and-build sponsors, chosen from a state-by-state research list and vetted for permission to crawl and for dated
pages. It turns each deal announcement since 2015 into a record (buyer, target, transaction type, the target's
business, sector and niche, state, status, disclosure gaps), each value next to the sentence it came from. On top of
the records it rebuilds each sponsor's **portfolio ledger** from its own announcements: entry dates come from the
investment release or from passing sentences ("ACP acquired StenTech in May of 2022"), renames are linked, and
add-ons show a company is still owned. From the ledger and the deals it builds four sourcing views: a **consolidation
map** of add-ons per niche with momentum, an **exit watch** of companies held four or more years with no sale
announced, **competing buyers** per niche and region, and a downloadable **sourcing brief**.

**Scope and method.** 1,795 articles from sixteen sponsors (2015 to November 2026), 1,048 unique deal announcements,
537 holdings in the ledger. Pages are fetched politely (robots.txt and crawl delays obeyed, honest user agent) and
saved, so the tool runs offline on cached data and says so. Extraction uses transparent rules, not a black-box model:
every value can be traced, runs are free and repeatable.

**Key findings (from the announcements, not the market).** Since 2022 the most-consolidated niches in the five
sectors are landscaping & grounds, engineering & consulting, HVAC & mechanical and managed IT services. The 85
companies with a dated entry and a sale were held a median of 5.6 years (middle half 3.7-7.4), so the exit watch
starts at four years; it lists 19 companies, among them E Source and Marco Sealing Solutions (Align), Mechanix Wear,
Ncontracts and Right Time Group (Gryphon), KGM (Compass) and PWSG (CID). In managed IT services, Goldner Hawn's Cyber
Advisors and Tonka Bay's Corporate Technologies were the most active buyers.

**Validation.** Accuracy was measured in three rounds of articles labelled from the source pages after the rules
were frozen in git, the last two labelled blind by a separate model. On the latest round (44 articles), deal
detection was 95%, publication dates 100%, transaction type 83%, buyer 81% and target 89%. **Sector and niche were
weak: about 50% and 30%.** Each new round scored lower than the one before, so the tuned numbers overstate how the
rules do on new sponsors. The app shows this beside the sector filter. Consistency checks find no impossible
records, and two runs give identical output. No label has yet been checked by a person.

**Business value.** A weekly sourcing scan that takes hours becomes minutes, with citations an investment committee can
check: who is rolling up a niche and how fast, which sponsor-owned companies are due to trade, and who would bid.
Because sectors are often wrong, the views are a first cut. A reviewer still checks each company's description before
outreach.

**Limitations.** Sixteen sponsors' self-published news is not the PE market. "No exit announced" may mean a quiet
sale. Prices and closing dates are almost never public. Companies no sponsor has bought are invisible here. Sector and
niche classification needs a better method; the next step is a language-model classifier for those two fields,
scored on the same held-out rounds.

**Creative features.** A portfolio ledger and exit watch rebuilt from press releases, a consolidation map with
momentum, competing buyers by niche and region, a sourcing brief generator, evidence cards with similar-deal search,
a new-deal digest with polite live refresh, ownership and exits inferred across announcements, and a disclosure-gap
report.
