# Personal Recommender: implementation work packages and decision records

> Scope update, 2026-11-06: the initial use case is Disney Theatrical stage productions and musicals. [theatrical-plan.md](theatrical-plan.md) governs the initial implementation and data plan. The long-form video tasks, MovieLens path, and implementation sequence below remain references for extension research.

Date: 2026-11-06. This is a planning document, not a record of completed implementation. See the [architecture and algorithm plan](design-plan.md) for the main design.

## Work packages

Each phase ends with reviewable artifacts and acceptance criteria. Fixed schedules are not estimated before data and compute conditions are known.

| Phase | Work | Deliverables | Acceptance / progression criteria |
| --- | --- | --- | --- |
| P0 Compatibility and data boundaries | Confirm the initial dataset and hardware; validate RecTools/PyTorch in isolation; check content usage terms | Dependency lock, data manifest, resource baseline | Reproducible minimal CPU example without using the global Python environment |
| P1 Data foundation | Contracts, catalog/event ingestion, temporal snapshots, TaskSpec, ID/OOV handling, metrics | Replayable research dataset and CLI | Manually verifiable examples; future events and features unavailable at the time are blocked |
| P2 Baselines | Popularity/content, ALS/BPR, EASE; RecVAE as a challenger | Quality/cost report under a fixed temporal protocol | Identical catalogs and relevance definitions; numerical and metric checks against reference implementations |
| P3 Sequences | SASRec, losses compared on the same architecture, time inputs, long/short-history controls | Sequence factorial experiments | Separate within-series continuation from cross-series discovery; no padding or causal-mask leakage |
| P4 Hybrid cold start | Content tower, ID residual, support gate, ID-dropout; optional request gate | Reports by cold/warm content and user slices | Zero-support ID residuals disabled; content-only path generates vectors independently |
| P5 Complete pipeline | Multiple candidate routes, RRF, eligibility rechecks, MMR, fallback | End-to-end CLI/API, bundle, and demo | Handle cold start, timeouts, and no eligible content; verifiable atomic rollback |
| P6 Real logs | Requests, actual impressions, mature labels, out-of-time reranker | Product-task data and offline replay | No fabricated logs; explicit label windows, denominators, and missing-data rules |
| P7 Online validation | Calibration, primary metrics, experiment assignment, exploration budget, guardrails | Controlled-experiment report or explicit readiness checklist | Claim online effects only with real traffic |
| P8 Frontier challengers | HSTU, eSASRec, optional Semantic-ID or page models | Pareto comparison under equal budgets | Adopt only when quality, coverage, or cost improves on or complements strong existing baselines |

In this original extension sequence, P1–P3 form the first implementation batch, P4–P5 the second, and P6 onward depends on real data and traffic. MovieLens can support research prototypes for P1–P5, but synthesizing impressions from ratings cannot satisfy P6 acceptance criteria.

## Experiment matrix

| Hypothesis | Controls | Confounders to exclude |
| --- | --- | --- |
| Content improves new-content performance | ID-only / content-only / hybrid | Future behavior for new content; pretraining on test material |
| Order adds value | History set / ordered model with identical inputs | Different parameter counts, losses, or training budgets |
| Time adds value | Same sequence model with/without actual time inputs | Target future timestamps or feedback used as inputs |
| Long and short interests complement each other | Long history / short history / fixed fusion / request gate | Benefits limited to active users; different long-history compute |
| ID-dropout helps | Identical hybrid with/without dropout | Other regularization or training steps changed simultaneously |
| Multiple routes add value | Single route / union / RRF | Different candidate budgets; low-quality additions that only increase coverage |
| Learned ranking adds value | RRF / reranker on fixed candidates | Retriever generated training candidates from examples it had already seen |
| Modalities contribute independently | Metadata+text / +visual / +audio | Promotional exposure, genre, cast, and missingness patterns |
| Large models are worth adopting | Strong SASRec / eSASRec / HSTU | Different sampling, sequence lengths, hardware, or tuning budgets |

## Initial decision records

| Decision | Choice | Revisit when |
| --- | --- | --- |
| Code organization | Modular monolith with separate training jobs | Independent scaling or failure isolation is demonstrably needed |
| Platform boundary | Own contracts, replay, and evaluation; upstream adapters | Keep external library defaults from defining business tasks |
| Retrieval | Exact first; ANN as a replaceable implementation | Measurements on fixed hardware exceed the performance budget |
| User model | Start with simple long-term pooling and a standard sequence model | Additional interest/time modeling shows independent gains |
| Item model | Content works independently; ID is a supported residual | Cold/warm tradeoffs or data requirements change |
| Ranking | Start with RRF; learn ranking when labels permit | Real, mature feedback has sufficient coverage |
| Multimodality | Text/metadata first; validate visual/audio inputs separately | Materials, compute, and backtest evidence support adoption |
| Project separation | Maintain the personal recommendation repository independently | Prevent implicit dependencies through brand-project data or code |

## Initial acceptance examples

1. The same profile/timestamp/bundle reproduces candidates and policy outcomes; random components record their seed.
2. Recall/NDCG can be checked manually on toy data, and empty-target behavior is explicit. Sampled metrics are not labeled full-catalog metrics.
3. Unseen item IDs can be scored through content; zero-history users receive an explainable fallback.
4. Continue Watching does not wrongly filter unfinished content, and Next Episode respects ordering and availability.
5. Features, ID support counts, and graph edges after the training cutoff cannot enter that training run. State replay accepts only events available at the time.
6. Incompatible models, embeddings, and indexes cannot load together. Timeout fallbacks still enforce eligibility constraints.

Prioritize these failure modes in tests rather than writing tests that simply mirror functions wrapping upstream calls.
