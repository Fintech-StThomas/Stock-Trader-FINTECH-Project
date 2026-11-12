# PE Deal Sourcing: validation

Accuracy per field against articles labelled from the source pages. Each held-out round was labelled after the rules were frozen in git; the newest round measures the rules as they are now and is the number to quote. Rules changed after a round's mismatches were read make that round in-sample, so its score at the freeze is kept next to its score now.

- **held-out-3**: round 3: 44 articles, any sponsor, labelled after the rules were frozen in commit 1835b7f (44 matched)
- **held-out-2**: round 2: 60 articles from the twelve added sponsors and 2015-2022, labelled after the freeze in commit 0bf8a14 (60 matched)
- **held-out**: round 1: 38 articles from the first four sponsors, labelled after the first freeze (38 matched)
- **development**: the articles the first rules were written on (in-sample) (71 matched)

| Field | **held-out-3 (quote this)** | held-out-2 at its freeze | held-out-2 now | held-out now | development now |
|---|---|---|---|---|---|
| is_deal | 42/44 (95%) | 60/60 (100%) | 60/60 (100%) | 38/38 (100%) | 71/71 (100%) |
| news_category | 40/44 (91%) | 58/60 (97%) | 58/60 (97%) | 38/38 (100%) | 71/71 (100%) |
| published_date | 44/44 (100%) | 60/60 (100%) | 60/60 (100%) | 38/38 (100%) | 71/71 (100%) |
| transaction_type | 30/36 (83%) | 43/48 (90%) | 45/48 (94%) | 30/30 (100%) | 54/54 (100%) |
| buyer | 29/36 (81%) | 44/48 (92%) | 45/48 (94%) | 30/30 (100%) | 54/54 (100%) |
| target | 32/36 (89%) | 45/48 (94%) | 45/48 (94%) | 30/30 (100%) | 53/54 (98%) |
| target_industry | 18/36 (50%) | 26/47 (55%) | 39/47 (83%) | 27/30 (90%) | 50/53 (94%) |
| target_state | 25/36 (69%) | 40/48 (83%) | 44/48 (92%) | 29/30 (97%) | 52/54 (96%) |
| deal_status | 27/36 (75%) | 42/48 (88%) | 42/48 (88%) | 29/30 (97%) | 54/54 (100%) |
| price_disclosure | 33/36 (92%) | 46/48 (96%) | 46/48 (96%) | 30/30 (100%) | 54/54 (100%) |
| focus_sector | 18/36 (50%) | 30/47 (64%) | 39/47 (83%) | — | — |
| niche | 8/27 (30%) | 13/31 (42%) | 18/31 (58%) | — | — |

95% Wilson intervals, held-out-3: is_deal 85%-99%; news_category 79%-96%; published_date 92%-100%; transaction_type 68%-92%; buyer 65%-90%; target 75%-96%; target_industry 34%-66%; target_state 53%-82%; deal_status 59%-86%; price_disclosure 78%-97%; focus_sector 34%-66%; niche 16%-48%

Deal vs other news (held-out-3): precision 97%, recall 97% (35 deals found, 1 false alarms, 1 missed).

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
| platform or add-on with no buyer and no target | 0 |

Repeatability: two builds from the same saved pages are identical (fingerprint `dc53e4d50f3ed47d`).

## Mismatches, held-out-3 (101)

| Article | Field | Label | Program |
|---|---|---|---|
| Huron Capital Completes Equity Investment in WD Lab Grown Diamonds | target_industry | Industrial manufacturing | Healthcare |
| Huron Capital Completes Equity Investment in WD Lab Grown Diamonds | focus_sector | Industrial manufacturing | Outside focus |
| Huron Capital Completes Equity Investment in WD Lab Grown Diamonds | niche | Other: Industrial manufacturing |  |
| Azalea Capital Invests in The Muffin Mam Inc. | target_industry | Consumer & food | Industrial manufacturing |
| Azalea Capital Invests in The Muffin Mam Inc. | deal_status | Completed | Unknown |
| Azalea Capital Invests in The Muffin Mam Inc. | focus_sector | Outside focus | Industrial manufacturing |
| Goldner Hawn & Tonka Bay Announce Sale of Quest Events | target_industry | Business services | Transportation & logistics |
| Goldner Hawn & Tonka Bay Announce Sale of Quest Events | focus_sector | Tech-enabled business services | Outside focus |
| Goldner Hawn & Tonka Bay Announce Sale of Quest Events | niche | Event & audiovisual services |  |
| Fort Point Moves up with 3Phase Elevator | is_deal | yes | False |
| Fort Point Moves up with 3Phase Elevator | news_category | Deal | Personnel |
| Fort Point Moves up with 3Phase Elevator | transaction_type | Platform investment |  |
| Fort Point Moves up with 3Phase Elevator | buyer | Fort Point Capital |  |
| Fort Point Moves up with 3Phase Elevator | target | 3Phase Elevator |  |
| Fort Point Moves up with 3Phase Elevator | target_industry | Building & infrastructure services |  |
| Fort Point Moves up with 3Phase Elevator | target_state | MA |  |
| Fort Point Moves up with 3Phase Elevator | deal_status | Completed |  |
| Fort Point Moves up with 3Phase Elevator | price_disclosure | Not stated |  |
| Fort Point Moves up with 3Phase Elevator | focus_sector | Building & facility services |  |
| Fort Point Moves up with 3Phase Elevator | niche | Elevator services |  |
| NuSource acquires Superior Safe & Security | target_industry | Building & infrastructure services | Business services |
| NuSource acquires Superior Safe & Security | focus_sector | Building & facility services | Tech-enabled business services |
| NuSource acquires Superior Safe & Security | niche | Facility services | Investigations & risk services |
| Gridiron Capital Partners with Health Monitor Network | target_industry | Business services | Healthcare |
| Gridiron Capital Partners with Health Monitor Network | price_disclosure | Not stated | Not disclosed (stated) |
| Gridiron Capital Partners with Health Monitor Network | focus_sector | Tech-enabled business services | Outside focus |
| Gridiron Capital Partners with Health Monitor Network | niche | Marketing & communications services |  |
| NRG Controls Announces Strategic Acquisition of Integrated Systems, In | target_state | GA |  |
| NRG Controls Announces Strategic Acquisition of Integrated Systems, In | deal_status | Completed | Unknown |
| Team Drive-Away Establishes Growth Focused Executive Team | news_category | Personnel | Fund / firm |
| Mason Wells Announces Sale of L.B. White | buyer | Modine Manufacturing Company | Unknown |
| Mason Wells Announces Sale of L.B. White | deal_status | Completed | Unknown |
| Montage Partners Acquires Southwest Data Products | niche | Electronics & electrical equipment | Other: Industrial manufacturing |
| Compass Group Invests in Critical Infrastructure Distributor | niche | Seals & industrial MRO | Other: Value-added distribution |
| Cleartelligence Accelerates Growth with Recruitment of Owen Frivold as | target_industry | IT services & cybersecurity | Business services |
| Cleartelligence Accelerates Growth with Recruitment of Owen Frivold as | focus_sector | IT services & cybersecurity | Tech-enabled business services |
| Cleartelligence Accelerates Growth with Recruitment of Owen Frivold as | niche | Other: IT services & cybersecurity | Engineering & consulting |
| Huron Capital’s High Street Insurance Partners Expands Northeast Prese | target_state | none | PA |
| Resilience Capital Partners Acquires Firequip | transaction_type | Platform investment | Add-on acquisition |
| Resilience Capital Partners Acquires Firequip | niche | Seals & industrial MRO | Other: Value-added distribution |
| SEAM Group Expands Services with Acquisition of EESCO | target_industry | Business services | Energy & environmental |
| SEAM Group Expands Services with Acquisition of EESCO | deal_status | Completed | Unknown |
| SEAM Group Expands Services with Acquisition of EESCO | focus_sector | Tech-enabled business services | Outside focus |
| SEAM Group Expands Services with Acquisition of EESCO | niche | Testing, inspection & calibration |  |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | transaction_type | Platform investment | Add-on acquisition |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | target | National Label (assets) | Assets of Global Package Printing Leader National Label |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | target_industry | Industrial manufacturing | Business services |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | target_state | PA |  |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | deal_status | Completed | Unknown |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | price_disclosure | Not stated | Not disclosed (stated) |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | focus_sector | Industrial manufacturing | Tech-enabled business services |
| Lux Global Label Company, Backed by Resilience Capital Partners, Acqui | niche | Packaging & plastics | Other: Tech-enabled business services |
| Gryphon Portfolio Company CORA Health Services Moves into Tennessee, K | buyer | CORA Health Services | Tennessee and Kentucky |
| Gryphon Portfolio Company CORA Health Services Moves into Tennessee, K | target | Champion Physical Therapy; Professional Therapies of Roanoke | Champion Physical Therapy |
| Gryphon Portfolio Company CORA Health Services Moves into Tennessee, K | deal_status | Completed | Unknown |
| Gryphon Investors Announces Close of Gryphon IV at over $1 Billion in  | is_deal | no | True |
| Gryphon Investors Announces Close of Gryphon IV at over $1 Billion in  | news_category | Fund / firm | Deal |
| Mason Wells Acquires Onyx Brands | target_industry | Consumer & food | Industrial manufacturing |
| Mason Wells Acquires Onyx Brands | target_state | AR |  |
| Mason Wells Acquires Onyx Brands | focus_sector | Outside focus | Industrial manufacturing |
| Pike Street Capital Announces Investment in DigiStor, Leading Provider | target_industry | Industrial manufacturing | Software & data |
| Pike Street Capital Announces Investment in DigiStor, Leading Provider | focus_sector | Industrial manufacturing | Outside focus |
| Pike Street Capital Announces Investment in DigiStor, Leading Provider | niche | Electronics & electrical equipment |  |
| AQUA Dermatology Announces Partnership with First Coast Mohs | target_state | FL |  |
| Huron Capital’s Aftermarket Platform Announces Acquisition of Rago Fab | transaction_type | Add-on acquisition | Platform investment |
| Huron Capital’s Aftermarket Platform Announces Acquisition of Rago Fab | buyer | Horsepower (Addictive Desert Designs and DV8 Offroad) | Huron Capital |
| Huron Capital’s Aftermarket Platform Announces Acquisition of Rago Fab | target_industry | Consumer & food | Industrial manufacturing |
| Huron Capital’s Aftermarket Platform Announces Acquisition of Rago Fab | focus_sector | Outside focus | Industrial manufacturing |
| Shermco Industries Acquires KTR Associates, a Power System Engineering | target_industry | Business services | Energy & environmental |
| Shermco Industries Acquires KTR Associates, a Power System Engineering | focus_sector | Tech-enabled business services | Outside focus |
| Shermco Industries Acquires KTR Associates, a Power System Engineering | niche | Engineering & consulting |  |
| WilliamsMarston Acquires California Corporate Tax Consultancy, Moffett | niche | Tax & accounting services | Engineering & consulting |
| Huron Capital Acquires ExperiGreen and Launches North American Residen | target_industry | Building & infrastructure services | Aerospace & defense |
| Huron Capital Acquires ExperiGreen and Launches North American Residen | focus_sector | Building & facility services | Outside focus |
| Huron Capital Acquires ExperiGreen and Launches North American Residen | niche | Landscaping & grounds |  |
| LEARN Behavioral Partners with Indiana-Based Behavior Analysis Center  | target_state | IN |  |
| Jensen Hughes Names Marc Kaplan New President & Chief Operating Office | news_category | Personnel | Other |
| Compass Electronics Group Acquires Protech Global Solutions | target_state | TX |  |
| Resilience Capital Partners further expands UK Print Operations | target_industry | Business services | Unknown |
| Resilience Capital Partners further expands UK Print Operations | focus_sector | Tech-enabled business services | Outside focus |
| Resilience Capital Partners further expands UK Print Operations | niche | Marketing & communications services |  |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | transaction_type | Growth / minority investment | Add-on acquisition |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | buyer | Huron Capital; Salt Creek Capital | Flex Equity Makes Non-Control |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | target | Pacific Shoring | Construction Safety Company |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | target_industry | Industrial manufacturing | Building & infrastructure services |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | target_state | CA |  |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | deal_status | Completed | Unknown |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | focus_sector | Industrial manufacturing | Building & facility services |
| Huron Capital and Salt Creek Capital Announce Investment in Pacific Sh | niche | Other: Industrial manufacturing | Other: Building & facility services |
| Huron Capital’s InterVision Announces Sixth Add-on Acquisition | transaction_type | Add-on acquisition | Unclear |
| Huron Capital’s InterVision Announces Sixth Add-on Acquisition | buyer | InterVision | Unknown |
| Huron Capital’s InterVision Announces Sixth Add-on Acquisition | target_industry | Software & data | Industrial manufacturing |
| Huron Capital’s InterVision Announces Sixth Add-on Acquisition | target_state | CA |  |
| Huron Capital’s InterVision Announces Sixth Add-on Acquisition | focus_sector | Outside focus | Industrial manufacturing |
| Oridian Capital Partners-Backed ACR Acquires RediBagUSA | buyer | ACR (AmerCareRoyal, LLC) | ACR |
| Oridian Capital Partners-Backed ACR Acquires RediBagUSA | target_industry | Value-added distribution | Consumer & food |
| Oridian Capital Partners-Backed ACR Acquires RediBagUSA | target_state | PA |  |
| Oridian Capital Partners-Backed ACR Acquires RediBagUSA | deal_status | Completed | Unknown |
| Oridian Capital Partners-Backed ACR Acquires RediBagUSA | focus_sector | Value-added distribution | Outside focus |
| Oridian Capital Partners-Backed ACR Acquires RediBagUSA | niche | Other: Value-added distribution |  |
| WilliamsMarston Acquires Valuation Consultant Oracle Capital | niche | Tax & accounting services | Engineering & consulting |

## Mismatches, held-out-2 (52)

| Article | Field | Label | Program |
|---|---|---|---|
| Gryphon Investors’ Mechanix Wear Acquires Chicago Protective Apparel | niche | Safety, marking & signage products | Other: Industrial manufacturing |
| ACP Completes Growth Capital Investment in WilliamsMarston LLC | niche | Tax & accounting services | Engineering & consulting |
| ShoreView Announces Sale of G.E.T. Enterprises, LLC | target_industry | Consumer & food | Value-added distribution |
| ShoreView Announces Sale of G.E.T. Enterprises, LLC | deal_status | Completed | Unknown |
| ShoreView Announces Sale of G.E.T. Enterprises, LLC | focus_sector | Outside focus | Value-added distribution |
| CID Capital Expands Investment Team with Addition of Tom Shaw | news_category | Personnel | Other |
| Meazure Learning Announces Acquisition Of Scantron’s Certification And | target | Scantron Certification and Licensure business | Scantron Corporation’s Certification and Licensure |
| Meazure Learning Announces Acquisition Of Scantron’s Certification And | niche | Other: Tech-enabled business services | Testing, inspection & calibration |
| Gryphon Investors Makes Majority Investment in PestRoutes | price_disclosure | Not stated | Not disclosed (stated) |
| XLerate Group Completes Fifth Add-on Deal, Expands into Pennsylvania | target_industry | Business services | Unknown |
| XLerate Group Completes Fifth Add-on Deal, Expands into Pennsylvania | target_state | PA |  |
| XLerate Group Completes Fifth Add-on Deal, Expands into Pennsylvania | focus_sector | Tech-enabled business services | Outside focus |
| XLerate Group Completes Fifth Add-on Deal, Expands into Pennsylvania | niche | Other: Tech-enabled business services |  |
| Huron Capital Launches Sixteenth ExecFactor ® Platform and Completes I | transaction_type | Platform investment | Add-on acquisition |
| Huron Capital Launches Sixteenth ExecFactor ® Platform and Completes I | buyer | Huron Capital | High Street Partners |
| Huron Capital Launches Sixteenth ExecFactor ® Platform and Completes I | target | High Street Partners; Peterson McGregor and Associates | Peterson McGregor and Associates |
| Huron Capital Launches Sixteenth ExecFactor ® Platform and Completes I | deal_status | Completed | Unknown |
| Mason Wells Acquires The Cleveland Electric Laboratories Company | niche | Electronics & electrical equipment | Other: Industrial manufacturing |
| Fort Point Capital Invests in CellSite Solutions | niche | Other: Building & facility services | Telecom & utility infrastructure |
| MGS Acquires European-Based Medical Injection Molder Knudsen Plast A/S | deal_status | Completed | Unknown |
| MGS Acquires European-Based Medical Injection Molder Knudsen Plast A/S | niche | Packaging & plastics | Other: Industrial manufacturing |
| ShoreView Invests in Hamilton Engineering, Inc. | target_industry | Business services | Unknown |
| ShoreView Invests in Hamilton Engineering, Inc. | focus_sector | Tech-enabled business services | Outside focus |
| ShoreView Invests in Hamilton Engineering, Inc. | niche | Engineering & consulting |  |
| Montage Partners Announces Investment in Military Fasteners | target_state | FL | AZ |
| Caylent Acquires Trek10 to Create the Most Comprehensive Dedicated AWS | niche | Managed IT services (MSP) | Other: IT services & cybersecurity |
| Provider of Real Estate and Employee Relocation Services Expands its P | transaction_type | Add-on acquisition | Platform investment |
| Provider of Real Estate and Employee Relocation Services Expands its P | buyer | Provider of real estate and employee relocation services (Sh | ShoreView Industries |
| Provider of Real Estate and Employee Relocation Services Expands its P | target_industry | Business services | Unknown |
| Provider of Real Estate and Employee Relocation Services Expands its P | focus_sector | Tech-enabled business services | Outside focus |
| Provider of Real Estate and Employee Relocation Services Expands its P | niche | Other: Tech-enabled business services |  |
| Gridiron Capital-Backed GSM Outdoors Acquires Gary Yamamoto Custom Bai | buyer | GSM Outdoors | Good Sportsman Marketing Outdoors |
| Gridiron Capital-Backed GSM Outdoors Acquires Gary Yamamoto Custom Bai | target_industry | Consumer & food | Industrial manufacturing |
| Gridiron Capital-Backed GSM Outdoors Acquires Gary Yamamoto Custom Bai | focus_sector | Outside focus | Industrial manufacturing |
| Burton Plumbing Services Completes Third Add-on Acquisition in 2021 | target_state | WI |  |
| Smile Brands and DecisionOne Dental Announce Strategic Partnership | transaction_type | Growth / minority investment | Add-on acquisition |
| Pike Street Names Sarah Karangelen as Finance and Operations Manager | news_category | Personnel | Other |
| Montage Partners Acquires Indo-European Foods, Inc. | target_industry | Consumer & food | Value-added distribution |
| Montage Partners Acquires Indo-European Foods, Inc. | focus_sector | Outside focus | Value-added distribution |
| Huron Capital’s Pueblo Mechanical & Controls Completes Second and Thir | price_disclosure | Not stated | Not disclosed (stated) |
| Tonka Bay Announces Sale of P.T. International | target_industry | Value-added distribution | Industrial manufacturing |
| Tonka Bay Announces Sale of P.T. International | focus_sector | Value-added distribution | Industrial manufacturing |
| Tonka Bay Announces Sale of P.T. International | niche | Seals & industrial MRO | Precision components & machining |
| ShoreView Announces Sale of Milamar Coatings, LLC | deal_status | Completed | Unknown |
| Quarterhill Announces Agreement to Acquire ETC | target | Electronic Transaction Consultants, LLC | ETC |
| Quarterhill Announces Agreement to Acquire ETC | target_state | TX |  |
| Align Capital Partners Realizes Investment in WilliamsMarston | deal_status | Completed | Unknown |
| Gridiron Capital Announces Sale of Engage2Excel | deal_status | Completed | Unknown |
| ShoreView Launches Test and Measurement Investment Strategy, Acquiring | niche | Electronics & electrical equipment | Precision components & machining |
| Jones Lake Management Acquires Water & Wetland | target_industry | Energy & environmental | Building & infrastructure services |
| Jones Lake Management Acquires Water & Wetland | focus_sector | Outside focus | Building & facility services |
| Gryphon Investors Sells MSD Ignition | niche | Electronics & electrical equipment | Precision components & machining |

## Mismatches, held-out (5)

| Article | Field | Label | Program |
|---|---|---|---|
| Align Partners with Engineering Services Firm AKS | target_state | OR |  |
| Counsel Press Expands into the Trial Courts with Case Anywhere Acquisi | target_industry | Software & data | IT services & cybersecurity |
| Marco Sealing Solutions Continues Expansion with Acquisition of Pride  | target_industry | Value-added distribution | Industrial manufacturing |
| Schneider Geospatial Expands Florida Presence | deal_status | Unknown | Completed |
| Huron Capital’s Platform 1-800 HANSONS Acquires Paramount Builders, LL | target_industry | Building & infrastructure services | Consumer & food |

## Notes

- Every labelled URL is in the dataset.
- Labels were made by reading each source page; see data/pe_deals/validation/README.md for who labelled what and how. The intervals are 95% Wilson intervals: with 30-50 labelled deals they are wide.
