# Disney Theatrical: initial use case and data plan

Date: 2026-11-06. The user has clarified that theatre means stage productions and musicals such as *The Lion King* and *Aladdin*. This document sets the current initial business plan for the personal recommendation project. Earlier long-form video plans remain reusable research references and no longer determine the initial implementation order. The project retains movie and music extension paths and remains independent of the brand project.

## 1. Recommendation task

The task extends from identifying preferred works to choosing a production and performance that suit the current travel and party context. The following is a proposed design, not a description of Disney's internal algorithms.

Use three entity levels:

- Work: the creative work and IP, including a musical's story, music, and themes.
- Production: a particular staging, such as a Broadway, West End, or touring production. Productions must not be merged solely by title.
- Performance: one showing of a production at a venue on a local date and time.

For ticketing, add a TicketOffer with price, fee scope, seating area, party size and adjacent-seat conditions, source, verification time, and expiry. No directly usable official open Disney inventory API has been verified. Weekly grosses and occupancy cannot establish live ticket availability.

## 2. Users and context

The first version forms request context from explicit user inputs: city or acceptable travel range, local date interval, budget and its per-person or whole-order scope, party size, party preferences, language, and viewing experience. Child-related or accessibility needs are used only when voluntarily provided, rather than inferred from movie preferences.

Model the purchaser, payer, and actual attendees separately. A four-person order does not mean four independent users expressed identical preferences. Refunds, transfers, and non-attendance do not necessarily indicate dislike. A musical and its namesake film may share IP or theme features, but film ratings must not be treated as stage-production ratings.

## 3. What public data can support

| Source | Confirmed content | Supported uses | What it cannot establish |
| --- | --- | --- | --- |
| Broadway League | Weekly grosses, admissions, performances, and occupancy | Seasonality, supply, and aggregate demand research | Individual preferences, impression conversion, or live ticket availability |
| IBDB | Productions, theatres, dates, casts, and creative teams | Entity identification, production relationships, and catalog structure | Actual user choices or personal recommendation effectiveness |
| CORGIS Broadway | Historical weekly aggregate CSV data | Method exploration within permitted terms | An official Disney user training set or real-time data |
| Official production and ticketing pages | Public descriptions and information that can be checked manually | A small prototype catalog with sources and timestamps, where permitted | A stable open API, rights to redistribute the full catalog, or verified live inventory |
| MovieLens | User movie ratings | Separate validation of general recommendation infrastructure and IP-link hypotheses | Disney musical user behavior or live-audience preferences |

Sources: [Broadway League](https://www.broadwayleague.com/research/grosses-broadway-nyc/), [IBDB](https://www.ibdb.com/about/), [CORGIS](https://corgis-edu.github.io/corgis/csv/broadway/). These sources cover Broadway rather than the full Disney global touring and West End catalog.

Broadway League explicitly restricts copying and reuse. Public readability does not imply permission for bulk ingestion. CORGIS downloads do not automatically resolve upstream data rights. Import data only after confirming the intended use is permitted, and keep data with unclear terms off GitHub.

This review did not identify a verified, publicly downloadable Disney Theatrical training set of individual purchases, impressions, or attendance feedback. The official Disney on Broadway homepage returned 403 during the review; no restrictions were bypassed and no data was collected from it.

## 4. Initial algorithm: preferences and feasibility

First, generate candidates at the Work/Production level using structured content features and explicitly stated preferences, such as musical style, themes, stage presentation, familiar IP, language, and viewing experience. A small catalog can use exact scoring without an ANN index or a large collaborative model.

Second, map candidates to Performances and check time, location, admission requirements, and travel feasibility. Check budget, party size, and seating constraints when a trusted ticketing source is available. Otherwise, mark these conditions as pending verification without guaranteeing purchase availability. Time zones, daylight saving time, and local dates must be explicit; filtering performances solely by UTC date is insufficient.

Third, produce an interpretable ranking based on preference match, schedule convenience, and budget fit, with controlled deduplication of performances of the same production. Initial scores represent rules and content matching rather than calibrated purchase probabilities. Users can revise preferences and inspect recommendation reasons.

For a limited Disney-only catalog, performance feasibility and decision explanations may matter more than complex work retrieval. Validate that hypothesis through user choice tests. A broader Broadway candidate set can be a separate experiment rather than an automatic scope expansion.

Once real choices and impressions are available, compare interpretable regression and simple ranking models before content/behavior hybrids. Evaluate ALS/BPR, VAE, SASRec, or HSTU only when repeat-purchase and sequence data are sufficient. Sequential models are not assumed optimal for infrequent theatre consumption.

## 5. Data collection and evaluation

### T0: catalog and constraint prototype

Build a small catalog with confirmed usage permission, explicit sources, and known information. Users enter preferences manually. Check performance dates, time zones, cities, budget scope, unknown inventory, and empty-candidate handling using manually verifiable scenarios. Synthetic data validates code, not recommendation effectiveness.

### T1: real preference validation

Collect real, explicit preferences in separately arranged user tests, such as candidate comparisons, favorites, and travel plans. Compare popularity, content similarity, contextual rules, and preference-plus-context methods. Measure recommendation acceptance, choice-task completion, explanation usefulness, and feasibility violations. Do not generalize a nonrepresentative sample to the overall market.

### T2: authorized behavior and ticketing integration

With appropriate authorization, integrate impressions, clicks, favorites, purchases, cancellations, attendance, and satisfaction logs. Define intent, purchase, and experience labels separately and handle observation windows. If real-time inventory exists, retain decision-time snapshots and verification timestamps. Without such an interface, keep recommendations subject to manual verification.

### T3: learned ranking and online experiments

Replay decisions in time order. Prevent leakage from orders or travel parties spanning training and test sets. Report Work-, Production-, and Performance-level outcomes so that multiple performances of the same production do not inflate hit rates. Compare learned ranking on real logs and compute uncertainty using appropriate independent units.

Initial success requires correct constraints, traceable catalogs, explainable recommendations, and better real-user choices than predefined controls. Aggregate box-office forecasting error is not a substitute for personal recommendation quality. Demand forecasting and brand audience targeting remain part of the other independent project.

## 6. Changes to earlier plans

- Prioritize T0→T1: stage-production entities, time and location constraints, explicit preferences, and content ranking.
- Use MovieLens for general model experiments and technical validation; do not describe it as real Disney Theatre data.
- Continue Watching, Next Episode, viewing duration, and movie-homepage metrics are not acceptance criteria for the first theatre version.
- Reuse data contracts, time-consistent replay, model bundles, explanation logs, and fallback design. Extend the catalog with productions, venues, performances, and ticketing entities.
- Compare traditional collaborative and sequential methods after real individual behavior becomes available. Confirm the data first rather than inventing interactions to fit a model roadmap.
