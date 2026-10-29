"""
The categories the extraction uses, defined in one place (the data dictionary, docs/PE_DEALS.md, prints them).

Every category is a list of word stems matched case-insensitively at word starts. They are deliberately plain, so a
reviewer can see why a record got its label (the matched stems are kept as evidence) and argue with it.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------- news categories (is it a deal?)
NEWS_CATEGORIES = {
    "Deal": "a transaction: an investment, acquisition, sale, recapitalization or combination",
    "Personnel": "hires, promotions, appointments, board changes",
    "Fund / firm": "fund closes, new offices, firm milestones, anniversaries",
    "Recognition / media": "awards, rankings, podcasts, interviews, articles, events",
    "Other": "anything else (hiring notices, thought leadership, community news)",
}

# ----------------------------------------------------------------------------- transaction types
TRANSACTION_TYPES = {
    "Platform investment": "the publishing firm itself acquires or invests in a company (a new platform or a control "
                           "recapitalization of a founder-owned business)",
    "Add-on acquisition": "a company the firm already owns (a portfolio / platform / partner company) acquires another",
    "Growth / minority investment": "the firm invests without taking control (minority, growth equity, structured capital)",
    "Merger / combination": "two companies join forces or merge, at least one backed by the firm",
    "Exit": "the firm (or a fund it manages) sells a portfolio company, or a portfolio company is acquired",
    "Dividend recapitalization": "a portfolio company refinances to return capital to its owners (no new owner)",
    "Unclear": "the text describes a transaction but its type cannot be read reliably",
}

DEAL_STATUS = {
    "Completed": "the text says the transaction has happened (has acquired, completed, closed, has invested)",
    "Announced / pending": "signed or agreed but not stated as closed (definitive agreement, will acquire, expected to close)",
    "Unknown": "neither is stated",
}

# ----------------------------------------------------------------------------- target industry (the target's business)
INDUSTRIES = {
    "Industrial manufacturing": ["manufactur", "fabricat", "machin", "precision", "component", "bearing", "coating",
                                 "molding", "moulding", "metal", "casting", "forging", "stamping", "engineered product",
                                 "assembl", "industrial product", "equipment maker", "plastics", "packaging", "welding",
                                 "electrical product", "power distribution", "tooling", "instrument", "adhesive",
                                 "sealant", "sealing product", "chemical", "formulat", "stencil", "erosion control product",
                                 "relays", "filtration product", "engineered", "power solution", "power equipment",
                                 "generator", "trench safety"],
    "Value-added distribution": ["distribut", "wholesale", "supply house", "fulfillment", "kitting"],
    "Aerospace & defense": ["aerospace", "defense", "aircraft", "aviation", "military", "space"],
    "IT services & cybersecurity": ["managed service", "managed it", "it service", "it solution", "msp", "cybersecurity",
                                    "cyber security", "information security", "network", "cloud", "it support",
                                    "technology service", "technology consult", "it consult", "it managed service", "desktop", "broadband",
                                    "help desk", "it infrastructure", "data center", "voip", "unified communication"],
    "Software & data": ["software", "saas", "analytics", "data platform", "digital platform", "application", "gis",
                        "geospatial", "govtech", "analysis solution", "property record"],
    "Business services": ["staffing", "consult", "outsourc", "collection", "receivable", "screening", "background check",
                          "marketing", "advertising", "printing", "document", "payroll", "insurance agen", "brokerage",
                          "testing", "inspection", "certification", "compliance", "training", "engineering service",
                          "professional service", "business service", "facility service", "janitorial", "security",
                          "translation", "event", "audio visual", "audiovisual", "credit", "advisory", "investigat",
                          "legal", "appellate", "engineering firm", "engineering consult", "engineering service",
                          "civil engineering", "surveying", "risk management", "research", "tax", "accounting",
                          "cost segregation", "calibration", "receivables management"],
    "Healthcare": ["healthcare", "health care", "medical", "dental", "dermatolog", "clinic", "physician", "patient",
                   "pharma", "hospital", "behavioral", "therapy", "veterinar", "revenue cycle", "laborator", "medical diagnostic",
                   "diagnostic testing", "diagnostic imaging", "point-of-care", "toxicology", "drug testing", "drug screening", "substance abuse"],
    "Building & infrastructure services": ["construction", "hvac", "plumbing", "roofing", "electrical contract",
                                           "electrical service", "electrical renovation", "electrician", "building product",
                                           "infrastructure services", "underground infrastructure", "utility services",
                                           "utility construction", "landscap", "fire protection", "elevator",
                                           "restoration", "commercial services", "mechanical contract", "mechanical service", "lawn",
                                           "pest control", "paving", "asphalt", "building envelope", "waterproofing",
                                           "building automation", "building control"],
    "Transportation & logistics": ["logistics", "trucking", "freight", "transportation", "fleet", "warehous",
                                   "towing", "delivery", "shipping"],
    "Consumer & food": ["consumer", "retail", "restaurant", "food", "beverage", "apparel", "produce", "pet", "e-commerce",
                        "direct-to-consumer", "resort", "fitness", "lighting", "home", "aftermarket automotive",
                        "automotive aftermarket", "camper", "animal health", "pet health", "pet wellness"],
    "Energy & environmental": ["energy", "environmental", "waste", "water treatment", "oil and gas", "oil & gas",
                               "oilfield", "petroleum", "renewable", "solar", "power generation", "power system",
                               "remediation", "recycling", "grid"],
    "Financial services": ["bank", "lender", "lending", "financial service", "wealth", "insurance carrier", "payment"],
}

# ----------------------------------------------------------------------------- customer end markets (not the target's industry)
END_MARKETS = {
    "Aerospace & defense": ["aerospace", "defense", "military", "aircraft"],
    "Automotive & vehicles": ["automotive", "vehicle", "truck", "aftermarket"],
    "Industrial / OEM": ["industrial", "oem", "original equipment manufacturer", "manufacturers"],
    "Distributors / channel": ["distributor", "dealer", "reseller"],
    "Healthcare": ["healthcare", "health care", "hospital", "medical", "patient"],
    "Government / public sector": ["government", "municipal", "public sector", "federal", "state agenc", "school"],
    "Construction & infrastructure": ["construction", "contractor", "infrastructure", "utility", "utilities"],
    "Energy": ["energy", "oil", "gas", "power", "renewable", "data center"],
    "Agriculture & food": ["agricultur", "farm", "food", "grocery"],
    "Consumer / retail": ["consumer", "retail", "homeowner", "residential"],
    "Commercial / SMB": ["small and mid", "small business", "mid-sized", "commercial customer", "businesses"],
    "Financial services": ["bank", "credit union", "financial institution", "lender"],
}

# ----------------------------------------------------------------------------- stated strategic rationale
RATIONALE_TAGS = {
    "Product / service expansion": ["product", "offering", "service line", "portfolio of", "solutions", "breadth",
                                    "suite", "complementary", "complement", "cross-sell", "full-service", "one-stop"],
    "Geographic reach": ["geograph", "footprint", "region", "presence in", "presence across", "national", "nationwide",
                         "coast", "midwest", "southeast", "northeast", "southwest", "mountain west", "new market",
                         "expands into", "expanding into", "location", "branch", "state of", "texas", "florida",
                         "international", "canada", "europe", "mexico"],
    "Customers / channels": ["customer", "client", "channel", "oem", "distributor", "end market", "market share",
                             "relationship", "account"],
    "Technology / capability": ["technolog", "capabilit", "expertise", "engineering", "automation", "proprietary",
                                "digital", "innovation", "certification", "know-how", "intellectual property"],
    "Capacity / scale": ["capacity", "scale", "facilit", "production", "throughput", "plant", "size", "larger"],
    "Talent / leadership": ["team", "talent", "leadership", "employees", "technicians", "engineers", "management"],
}

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
STATE_ABBR = {"Ala.": "AL", "Ariz.": "AZ", "Ark.": "AR", "Calif.": "CA", "Colo.": "CO", "Conn.": "CT", "Del.": "DE",
              "Fla.": "FL", "Ga.": "GA", "Ill.": "IL", "Ind.": "IN", "Kan.": "KS", "Ky.": "KY", "La.": "LA", "Md.": "MD",
              "Mass.": "MA", "Mich.": "MI", "Minn.": "MN", "Miss.": "MS", "Mo.": "MO", "Mont.": "MT", "Neb.": "NE",
              "Nev.": "NV", "N.H.": "NH", "N.J.": "NJ", "N.M.": "NM", "N.Y.": "NY", "N.C.": "NC", "N.D.": "ND",
              "Okla.": "OK", "Ore.": "OR", "Pa.": "PA", "R.I.": "RI", "S.C.": "SC", "S.D.": "SD", "Tenn.": "TN",
              "Vt.": "VT", "Va.": "VA", "Wash.": "WA", "W.Va.": "WV", "Wis.": "WI", "Wyo.": "WY"}
REGIONS = {
    "Midwest": {"IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"},
    "Northeast": {"CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"},
    "South": {"AL", "AR", "DE", "DC", "FL", "GA", "KY", "LA", "MD", "MS", "NC", "OK", "SC", "TN", "TX", "VA", "WV"},
    "West": {"AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"},
}


def region_of(state: str) -> str:
    for r, states in REGIONS.items():
        if state in states:
            return r
    return "Unknown"


# Cities that releases name without their state ("Based in Miami", "a Houston-based distributor"). Cities whose name is
# shared by large cities in two states (Kansas City, Portland, Springfield, Columbus ...) are left out on purpose: the
# location stays unknown rather than guessed.
CITY_STATE = {
    "New York City": "NY", "Los Angeles": "CA", "Chicago": "IL", "Houston": "TX", "Phoenix": "AZ", "Philadelphia": "PA",
    "San Antonio": "TX", "San Diego": "CA", "Dallas": "TX", "San Jose": "CA", "Austin": "TX", "Jacksonville": "FL",
    "Fort Worth": "TX", "Dallas-Fort Worth": "TX", "Indianapolis": "IN", "Charlotte": "NC", "San Francisco": "CA",
    "Seattle": "WA", "Denver": "CO", "Nashville": "TN", "Oklahoma City": "OK", "El Paso": "TX", "Boston": "MA",
    "Las Vegas": "NV", "Detroit": "MI", "Memphis": "TN", "Louisville": "KY", "Baltimore": "MD", "Milwaukee": "WI",
    "Albuquerque": "NM", "Tucson": "AZ", "Fresno": "CA", "Sacramento": "CA", "Atlanta": "GA", "Omaha": "NE",
    "Raleigh": "NC", "Miami": "FL", "Minneapolis": "MN", "Tulsa": "OK", "Cleveland": "OH", "Tampa": "FL",
    "New Orleans": "LA", "Pittsburgh": "PA", "Cincinnati": "OH", "St. Louis": "MO", "Orlando": "FL", "Salt Lake City": "UT",
    "St. Paul": "MN", "Saint Paul": "MN", "Scottsdale": "AZ", "Grand Rapids": "MI", "Des Moines": "IA", "Boise": "ID",
    "Richmond": "VA", "Birmingham": "AL", "Buffalo": "NY", "Rochester": "NY", "Syracuse": "NY", "Albany": "NY",
    "Hartford": "CT", "Providence": "RI", "Madison": "WI", "Green Bay": "WI", "Fargo": "ND", "Sioux Falls": "SD",
    "Little Rock": "AR", "Knoxville": "TN", "Chattanooga": "TN", "Charleston": "SC", "Greenville": "SC", "Toledo": "OH",
    "Akron": "OH", "Dayton": "OH", "Ann Arbor": "MI", "Lansing": "MI", "Wichita": "KS", "Lincoln": "NE",
}
AMBIGUOUS_CITIES = ("Kansas City", "Portland", "Springfield", "Columbus", "Arlington", "Aurora", "Columbia", "Jackson")
