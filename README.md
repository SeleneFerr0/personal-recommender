# Personal Theatre Recommender

A runnable Python research prototype that checks performance feasibility before ranking options using individual and party preferences. Designed for applications similar to Disney Theatrical, it **does not connect to Disney data, live ticket inventory, or real audience logs**. All example productions, prices, performances, and choice records are fictional.

Requires Python >=3.11. The core package and tests have no third-party runtime dependencies. This project is independent of the brand project.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
personal-rec demo
personal-rec demo --export-dir data/demo --output artifacts/demo.json
personal-rec recommend --catalog data/demo/catalog.json --request data/demo/request.json
python -m unittest discover -s tests -v
```

To run without installing the package: `PYTHONPATH=src python -m personal_recommender demo`. The project has been tested on Python 3.12. Your terminal must use a working Python 3.11+ interpreter rather than an older system Python.

## Implemented features

- Work → Production → Performance identifiers; budget scope, city, local dates, language, admission age, accessibility, party size, adjacent seats, and repeat-consumption rules.
- Ticket snapshots with provenance, verification time, and expiry. Strict mode excludes unknown critical conditions; `allow_unverified=true` returns explicitly pending-verification candidates without guaranteeing that tickets can be purchased. Cancelled performances do not become eligible again when a snapshot expires.
- Interpretable features for content match, aversion, price and time fit, familiarity, popularity, minimum and mean party preference match, and interaction terms.
- Four fixed baselines: popularity, content, context, and hybrid. Learning methods include pairwise logistic regression, conditional logit with an outside/no-choice option, validation-selected blend weights, and a `selected` ranking method chosen by validation NDCG.
- Work-level deduplication across performances and configurable simple diversity reranking. Results include raw scores, feature contributions, and rejection reasons. Scores are not purchase probabilities.
- A candidate-attribute disagreement heuristic for suggesting the next preference question. This is a simple research entry point; it does not implement an LLM or an experimentally validated optimal questioning policy.
- JSON model files with feature-order, hash, and time checks. Models trained on synthetic data cannot be applied to catalogs marked as real.

## Training and comparison

Generating synthetic data validates execution; it does not provide real Disney data:

```bash
personal-rec simulate --output-dir data/sim --n 360 --regime mixed
# Read the split times in data/sim/manifest.json, then run:
personal-rec compare --catalog data/sim/catalog.json \
  --observations data/sim/observations.json \
  --validation-at 2030-01-03T06:00:00+00:00 \
  --test-at 2030-01-03T21:00:00+00:00 \
  --epochs 160 --output artifacts/comparison.json --model-dir models/sim
personal-rec benchmark --seeds 7,19,43 --n 600 --epochs 180 \
  --output artifacts/stress-test.json
```

`compare` trains the learning methods and uses only validation data to choose the adaptive blend by log loss and the `selected` ranking method by NDCG, breaking ties by log loss. It retains all test results, including methods that do not win. `selected` can choose a simple baseline when it performs better; it is not guaranteed to win on the test set. Selection adds a validation search over seven candidate methods, with no additional fitting. Training, validation, and test sets are split by time, with cross-partition profiles and label windows purged. Test results do not influence model selection.

Evaluation uses the actually displayed choice sets in the input logs. NDCG and Hit are conditional on an item being chosen; choice log loss includes the no-choice option. Tied scores use the exact expected metric under random tie-breaking. Metrics are macro-averaged by profile, and difference intervals use paired profile bootstrap. This is neither a full-catalog retrieval benchmark nor a causal incrementality experiment. The CLI currently supports a research comparison workflow; an online experimentation service is not implemented.

## Importing your own data

See `examples/catalog.json` for the catalog format and `examples/request.json` for requests. Both files are explicitly marked synthetic. A real catalog should include its own `snapshot_at`, provenance, and ticket status known at that time. Do not use the present-day catalog for historical requests.

`observations.json` is an array of objects with this structure:

```json
{
  "request": {"...": "Complete Request; see examples/request.json for the structure"},
  "shown_ids": ["an actually displayed, eligible performance_id"],
  "chosen_id": null,
  "observed_at": "2030-01-03T12:05:00+00:00"
}
```

`chosen_id=null` means no choice within an explicitly defined observation window. Missing or immature feedback must not be represented as no choice. Each request currently accepts one choice; buyer and attendee identities for multi-ticket orders must be modeled correctly upstream.

The current `compare` command consumes one as-of catalog snapshot applicable to all requests. When catalogs or inventory change frequently, process batches against their corresponding snapshots. Historical inventory version resolution, payment integration, travel distances, currency conversion, complete seat maps, and dedicated text or multimodal encoders are not implemented. Unknown critical facts, such as age, are not assumed to satisfy requirements. Requests with no feasible candidates return an empty list.

## Method limitations

Conditional logit assumes independence of irrelevant alternatives (IIA) and linear utility. Pairwise learning omits some competition within the list. The fixed hybrid supports a start without labels, but its weights are not calibrated. The adaptive method uses validation-selected blend weights rather than a dynamic per-person gate. These standard methods and their combinations do not establish novelty or universal superiority.

Private frontier research and internal benchmark results are stored outside the repository. Local historical research files are protected by Git ignore rules. Do not include personal data, ticket snapshots, models, research attachments, or private reports in release packages.
