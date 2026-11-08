"""Shared eligibility and interpretable feature pipeline for every algorithm."""
from __future__ import annotations

from dataclasses import asdict
import math
from zoneinfo import ZoneInfo
from .schema import Catalog, Performance, Request, Show, instant

FEATURES = ("bias", "content", "aversion", "price_fit", "time_fit", "familiarity",
            "popularity", "group_min", "group_mean", "content_price", "content_time", "verified")
WEIGHTS = {
    "popularity": (0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0),
    "content": (0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "context": (0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0),
    "hybrid": (0, 1.5, 2, 1, 1, .3, .3, 1, .5, 0, 0, .3),
}


# Fixed nonlinear basis for the residual choice model: hinge splines on continuous
# features plus a few cross terms. Order is part of the saved model contract.
HINGE_FEATURES = ("content", "price_fit", "time_fit", "popularity", "group_min", "content_time")
HINGE_KNOTS = (.25, .5, .75)
CROSS_FEATURES = (("price_fit", "time_fit"), ("content", "group_min"), ("price_fit", "group_min"),
                  ("content", "popularity"), ("time_fit", "group_min"))
BASIS = tuple(f"{f}>{k}" for f in HINGE_FEATURES for k in HINGE_KNOTS) + tuple(f"{a}*{b}" for a, b in CROSS_FEATURES)
_IDX = {f: i for i, f in enumerate(FEATURES)}
LEARNED = ("pairwise", "choice", "residual", "adaptive", "selected")


def basis(x):
    return [max(0., x[_IDX[f]] - k) for f in HINGE_FEATURES for k in HINGE_KNOTS] + \
           [x[_IDX[a]] * x[_IDX[b]] for a, b in CROSS_FEATURES]


def residual(model, x, work_id=None):
    """Nonlinear basis and shrunk work effect; zero for models without them."""
    extra = getattr(model, "extra", None) or {}
    value = dot(extra["basis_weights"], basis(x)) if extra.get("basis_weights") else 0.
    return value + extra.get("work_bias", {}).get(work_id, 0.)


def eligibility(show: Show, perf: Performance, request: Request):
    """Unknown critical facts never turn into a claim of confirmed bookability."""
    now = instant(request.at)
    reasons, unknown = [], []
    if instant(perf.known_at) > now:
        return ["not_known_at_request"], []
    if instant(perf.starts_at) <= now:
        reasons.append("already_started")
    local = instant(perf.starts_at).astimezone(ZoneInfo(perf.timezone))
    if not request.start_date <= local.date().isoformat() <= request.end_date:
        reasons.append("outside_local_date_window")
    if perf.city.casefold() not in {c.casefold() for c in request.cities}:
        reasons.append("city_mismatch")
    if request.languages and show.language not in request.languages:
        reasons.append("language_mismatch")
    if request.youngest_age is not None:
        if show.min_age is None:
            unknown.append("minimum_age")
        elif request.youngest_age < show.min_age:
            reasons.append("minimum_age")
    if not request.allow_repeats and show.work_id in request.seen_work_ids:
        reasons.append("repeat_not_allowed")
    # A record whose snapshot lies in the future is unavailable, not stale usable data.
    if perf.verified_at and instant(perf.verified_at) > now:
        return reasons + ["future_ticket_snapshot"], []
    if perf.status == "cancelled" and (not perf.verified_at or instant(perf.verified_at) <= now):
        reasons.append("cancelled")
    fresh = bool(perf.verified_at and now < instant(perf.expires_at))
    if fresh:
        if perf.status in {"cancelled", "sold_out"}:
            reasons.append(perf.status)
        if perf.status != "available":
            unknown.append("inventory_status")
        if perf.price is None or perf.currency != request.currency:
            unknown.append("price_or_currency")
        elif perf.price > request.per_person_budget:
            reasons.append("over_budget")
        if perf.available_seats is None:
            unknown.append("seat_count")
        elif perf.available_seats < request.party_size:
            reasons.append("insufficient_seats")
        if request.require_consecutive:
            if perf.consecutive_seats is None:
                unknown.append("consecutive_seats")
            elif perf.consecutive_seats < request.party_size:
                reasons.append("insufficient_consecutive_seats")
        if request.requires_accessibility:
            if perf.accessible is None:
                unknown.append("accessibility")
            elif not perf.accessible:
                reasons.append("accessibility_mismatch")
    else:
        unknown.append("missing_or_expired_ticket_snapshot")
    if unknown and not request.allow_unverified:
        reasons.append("requires_verification")
    return reasons, unknown


def affinity(preferences, tags):
    positive = sum(max(0, v) for v in preferences.values())
    return sum(max(0, preferences.get(t, 0)) for t in set(tags)) / positive if positive else 0.0


def features(show, perf, request):
    content = affinity(request.preferences, show.tags)
    neg_total = sum(-v for v in request.preferences.values() if v < 0)
    aversion = -sum(-request.preferences.get(t, 0) for t in set(show.tags) if request.preferences.get(t, 0) < 0) / neg_total if neg_total else 0
    if set(show.tags) & set(request.avoid_tags):
        aversion = -1.0  # avoid tags are soft aversions, never undocumented hard filters
    _, unknown = eligibility(show, perf, request)
    fresh = bool(perf.verified_at and instant(perf.verified_at) <= instant(request.at) < instant(perf.expires_at))
    budget = request.per_person_budget
    price = max(0, 1 - perf.price / budget) if fresh and perf.price is not None and perf.currency == request.currency and budget > 0 else 0
    local = instant(perf.starts_at).astimezone(ZoneInfo(perf.timezone))
    hour = local.hour + local.minute/60
    delta = abs(hour - request.preferred_hour)
    time_fit = 1 - min(delta, 24 - delta) / 12
    member = [sum(p.get(t,0) for t in set(show.tags))/max(1e-12,sum(abs(v) for v in p.values()))
              for p in request.member_preferences] or [content]
    return [1., content, aversion, price, time_fit,
            float(show.work_id in request.seen_work_ids), show.popularity,
            min(member), sum(member)/len(member), content*price, content*time_fit, float(not unknown)]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def score(name, x, model=None, work_id=None):
    if name in WEIGHTS:
        return dot(WEIGHTS[name], x)
    if name in LEARNED:
        if model is None:
            raise ValueError("trained model required")
        if name != model.method:
            raise ValueError("algorithm must match saved model method")
        learned = dot(model.weights, x) + residual(model, x, work_id)
        if name == "adaptive":
            return model.blend*learned + (1-model.blend)*dot(WEIGHTS["hybrid"], x)
        return learned
    raise ValueError(f"unknown algorithm: {name}")


def candidates(catalog, request):
    if instant(catalog.snapshot_at) > instant(request.at):
        raise ValueError("catalog snapshot postdates request; use an as-of catalog")
    shows = {s.production_id: s for s in catalog.shows}
    accepted, rejected = [], []
    for perf in catalog.performances:
        show = shows[perf.production_id]
        reasons, unknown = eligibility(show, perf, request)
        if reasons:
            rejected.append({"performance_id": perf.performance_id, "reasons": reasons})
        else:
            accepted.append((show, perf, features(show, perf, request), unknown))
    return accepted, rejected


def recommend(catalog, request, algorithm="hybrid", model=None, k=5, diversity=.15, max_per_work=1):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be positive")
    if not math.isfinite(diversity) or not 0 <= diversity <= 1 or max_per_work < 1:
        raise ValueError("invalid diversity or work quota")
    if algorithm not in WEIGHTS and algorithm not in LEARNED:
        raise ValueError("unknown algorithm")
    if algorithm not in WEIGHTS and (model is None or model.method != algorithm):
        raise ValueError("matching trained model required")
    if model and model.metadata and model.metadata.get("last_observed_at") and instant(model.metadata["last_observed_at"]) > instant(request.at):
        raise ValueError("model trained on feedback after this request")
    if model and model.metadata and model.metadata.get("synthetic_training", False) and not catalog.synthetic:
        raise ValueError("synthetic-trained model is restricted to synthetic catalogs")
    rows, rejected = candidates(catalog, request)
    pool = [(score(algorithm, x, model, s.work_id), s, p, x, unknown) for s,p,x,unknown in rows]
    selected, count = [], {}
    while pool and len(selected) < k:
        eligible = [r for r in pool if count.get(r[1].work_id, 0) < max_per_work]
        if not eligible:
            break
        def utility(row):
            tags = set(row[1].tags)
            overlap = max((len(tags & set(v[1].tags))/max(1,len(tags | set(v[1].tags))) for v in selected), default=0)
            return row[0] - diversity*overlap
        best = sorted(eligible, key=lambda r: (-utility(r), r[2].performance_id))[0]
        selected.append(best)
        count[best[1].work_id] = count.get(best[1].work_id, 0)+1
        pool.remove(best)
    results = []
    for value, show, perf, x, unknown in selected:
        contributions = {f: w*v for f,w,v in zip(FEATURES, model.weights if model and algorithm in {"choice","pairwise","residual","selected"} else WEIGHTS["hybrid"] if algorithm=="adaptive" else WEIGHTS[algorithm], x)}
        if algorithm == "adaptive":
            contributions = {f: (model.blend*w+(1-model.blend)*h)*v for f,w,h,v in zip(FEATURES,model.weights,WEIGHTS["hybrid"],x)}
        if model is not None and getattr(model, "extra", None):
            extra = model.extra
            contributions["nonlinear_residual"] = dot(extra.get("basis_weights") or [0.]*len(BASIS), basis(x))
            contributions["work_effect"] = extra.get("work_bias", {}).get(show.work_id, 0.)
        results.append({"work_id": show.work_id, "production_id": show.production_id,
                        "title": show.title, "performance": asdict(perf), "score": value,
                        "bookability": "requires_verification" if unknown else "verified_snapshot",
                        "verification_gaps": unknown, "features": dict(zip(FEATURES,x)),
                        "score_contributions": contributions,
                        "explanation": {"matched_tags": sorted(t for t in show.tags if request.preferences.get(t,0)>0),
                                        "group_min_match": x[7], "score_is_purchase_probability": False}})
    return {"request_id": request.request_id, "algorithm": algorithm, "synthetic_data": catalog.synthetic,
            "model_id": model.model_id if model else None, "results": results, "rejected": rejected,
            "eligible_count": len(rows), "returned_count": len(results),
            "note": "Ticket snapshots are time-limited observations, not a booking guarantee."}
