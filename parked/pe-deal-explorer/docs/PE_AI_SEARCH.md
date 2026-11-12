# Lighthouse AI research setup

The primary PE screen is now AI Deal Research. One centered, 140-pixel-tall question box replaces the manual search, industry, firm, location, rationale and date filters. The stock workspace and Deal Detail remain available. Source coverage and quality tools remain in the repository for team review, but their navigation options are removed. The former explorer renderer remains in code for regression checks; it is not the public PE entry point.

Example question: Find technology acquisitions announced in the last 90 days. Show whatever is disclosed about the ownership stake, deal value, and the target's public or private status.

## What the assistant does

A server-side request to OpenAI's Responses API supplies the question, current date and up to 80 relevance-ranked saved PE records with short excerpts. The configured model selects relevant saved deals and can use web search for recent public evidence. Up to eight results show buyer, target, sector, geography, publication date, transaction status, ownership stake, deal value, rationale and dated listing evidence. Queries stating “recent” without a timeframe default to 90 days; the interpreted period is shown to the user.

All saved records are searched locally using lexical relevance, industry/firm hints and recency hints before the top 80 candidates are supplied to the model. This removes the first-200-record truncation. Each supplied excerpt is capped at 500 characters. The interface and downloads report the number scanned and supplied. Candidate retrieval is not a guarantee of exhaustive semantic matching; the model must still check the complete question and date/type restrictions.

No credentials are sent to the model in the prompt. The API key is used only in the server's authorization header. Public visitors cannot enter or retrieve the key. Search questions and selected public data are sent to OpenAI; the interface says so. Responses request `store=false`; this is not a claim that all provider retention is eliminated.

## Owner setup

The Streamlit deployment owner must install the updated requirements and add these settings in the app's server-side Secrets settings:

```toml
OPENAI_API_KEY = "YOUR_PROJECT_API_KEY"
OPENAI_MODEL = "gpt-6-astra"
PE_AI_WEB_SEARCH = true
PE_AI_DAILY_LIMIT = 100
```

Use an API project with access to the selected model, Responses API and web search. `OPENAI_MODEL` is configurable; the example/default uses the model in the current official Responses web-search documentation. Availability and billing must be verified in the actual project. The app does not connect through the user's ChatGPT browser login.

Do not paste a real key into chat, source code, the public sidebar, or a committed file. For local development, the same names may be set as environment variables or in ignored `.streamlit/secrets.toml`. Shell environment variables require no extra Python dependency. This app does not automatically load `.env` files.

Without a key, the screen honestly shows setup pending and does not present keyword matches or mocked responses as AI output. Offline mode refuses API calls. A real paid API query has not yet been run in this implementation environment. Verify it with a valid project credential before deploying.

## Evidence and financial interpretation

Saved records are selected by known IDs; returned buyer, target, status and other saved facts come from the original dataset, not rewritten model values. Unknown IDs are rejected. Additional model facts require supporting quotes whose URLs match the API web-search source/citation metadata, or whose quotes appear in supplied saved evidence. Uncited web facts are dropped.

Citation membership is not independent proof that the model quoted or interpreted a page correctly. Web facts remain AI-extracted, with review pending. The model is instructed to use exact quotes; team checks must verify them. This system does not fetch source pages again to independently validate every quote.

Undisclosed stake and deal value remain unknown. Majority does not imply 100%. Company valuation is not the amount paid. Announced acquisitions are not completed ownership transfers. Public/private listing status must have a source and supported as-of date, displayed beside the status; the assistant must not infer delisting from a take-private announcement. Dated evidence should not be presented as a verified present-day exchange status.

The former extraction accuracy report covers the historical rule-based dataset only. It does not measure this new AI query, web search, matching, or enrichment pipeline. A new evaluation should cover relevant/irrelevant matches, recent-date interpretation, minority stakes, valuations, announced take-privates, uncertain listing status, and unsupported source claims.

## Limits and costs

Search requires an explicit form submission. Normal reruns, navigation and downloads reuse session results. The app allows two concurrent requests, 20 submissions per visitor session, a 15-second cooldown and a configurable shared daily allowance per server process. Requests cap output tokens, use low reasoning effort for configured GPT-5/6 models and cap built-in web tool calls at three.

These are request controls, not a dollar spending cap. Restarts reset the in-memory shared allowance, and multiple processes each have their own allowance. Before broad public release, configure provider-side project limits/alerts as available and use durable rate limiting or authenticated access if needed. API errors use generic messages and never show raw response bodies or secrets. No automatic retry can silently multiply a paid request.

## Review and test

Run `python -m pytest tests/test_pe_agent.py tests/test_workspaces.py tests/test_pe_competitive.py tests/test_pe_quality.py tests/test_pe_explorer.py -q`.

Tests exercise structured requests, saved-record preservation, missing values, invalid IDs, citation filtering, listing dates, valuation handling, missing keys, offline behavior, rate limits, safe errors, session persistence, the centered search box and clickable prompt suggestions and stock/PE navigation. API responses are mocked. They establish application behavior, not the quality of real model answers.

Official implementation references:

- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/tools-web-search

Neither tests nor UI research results replace human validation of the class deliverable.


## Centered search and complete exports

The main screen has a centered search area up to 1,000 pixels wide, a 140-pixel multiline field, and six clickable suggested prompts. Suggestions fill the field without making a paid request; the visitor can edit and then submit. Sidebar navigation remains available without search filters.

The full deal CSV now includes original headline, all available result facts, original saved-record columns, saved evidence, review notes, per-field quotes/URLs, article titles, every source URL, retrieval scope and search context. A separate article CSV distinguishes supporting sources from other web pages consulted. The ZIP contains deals.csv, articles.csv, evidence.csv, the unmodified research JSON and a reading guide. Original article bodies are not fetched into exports: links open the original pages, and available supporting excerpts are included. Spreadsheet formula cells from external text are escaped in CSV; original strings remain in JSON.
