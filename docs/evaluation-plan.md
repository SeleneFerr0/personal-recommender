# Evaluation plan

This document defines a proposed experimental protocol. It does not report completed experiments or promise performance.

## Define the task first

| Task | Input | Output | Key slices |
| --- | --- | --- | --- |
| Long-term interests | History and context available at request time | Content list | New, low-activity, and active users |
| Next item | Ordered history and current session | Next-item candidates | Session length and time gaps |
| Similar content | Seed content and optional preferences | Similar, useful content | Genre, style, and novelty |
| New content | Metadata and media features available before release | Content or audience match scores | Zero or few interactions |

## Data protocol

- Use global time boundaries and rolling backtests. Define training, validation, test, and label observation windows explicitly. Treat user holdout as a separate experiment.
- Reconstruct content availability, regional and licensing conditions, age suitability, and explicit user blocks at request time.
- Build feature vocabularies, statistics, content representations, and sampling distributions within the permitted information boundary. Record external pretrained model versions and known knowledge time boundaries.
- Do not interpret unexposed content as an explicit dislike. Unclicked impressions also require attention to position, visibility, and observation windows.
- Use the same user set, candidate rules, K, relevance definition, and metric implementation across models. State the denominator of Recall@K precisely.

## Metrics by layer

- Retrieval: candidate Recall@K, coverage, and new-content coverage. Measure ANN recall against exact nearest neighbors separately.
- Ranking: NDCG@K and Recall@K; check calibration when predicting probabilities.
- Lists: diversity, novelty, repetition, and content concentration.
- Product: satisfaction, skips or quick exits, continued consumption, and retention. Clicks, duration, and completion rate are distinct signals and are not interchangeable.
- Engineering: end-to-end p50/p95/p99 latency, throughput, memory, per-request cost, and fallback behavior.

## Experiment sequence

1. Popularity and content similarity; SAR and implicit ALS/BPR.
2. Compare Multi-VAE and sequential models under a common protocol.
3. Add metadata, text, visual, and audio inputs incrementally, with ablation and cost comparisons.
4. Evaluate new content and changing interests before trying exploration policies and more complex fusion.
5. Run randomized controlled experiments when online testing becomes possible. Compute uncertainty at the independent assignment unit and define primary metrics and observation periods in advance.

Sampled evaluation is a clearly labeled supplementary experiment and must not be compared directly with full-catalog evaluation. Google research shows that sampled metrics can even change relative model rankings: [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/).

Before using counterfactual evaluation, check logging-policy probabilities, support, and estimator variance. It is not an unconditional substitute for A/B testing.

References for online iteration: [Netflix Interleaving](https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55), [Spotify personalization and experimentation responsibilities](https://engineering.atspotify.com/2026/1/why-we-use-separate-tech-stacks-for-personalization-and-experimentation).
