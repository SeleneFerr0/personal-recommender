"""Tie-aware single-choice metrics and validation-only model selection."""
from __future__ import annotations

from dataclasses import asdict
import math
import random
from .schema import instant
from .engine import score, WEIGHTS
from .learning import Model, design, fit, fit_residual, probabilities, shown_work_ids, temporal_split

# Validation grid for the residual model. "off" settings keep the linear model reachable, so
# the selector can fall back when the extra capacity does not pay for itself.
RESIDUAL_GRID = (
    {"use_basis": False, "use_work": False},
    {"use_basis": True, "use_work": False, "l2_basis": .01},
    {"use_basis": True, "use_work": False, "l2_basis": .05},
    {"use_basis": False, "use_work": True, "l2_work": .05},
    {"use_basis": True, "use_work": True, "l2_basis": .01, "l2_work": .05},
    {"use_basis": True, "use_work": True, "l2_basis": .05, "l2_work": .2},
)


def tie_metrics(scores, target, k):
    """Exact expected Hit/NDCG with uniform random order inside a score tie."""
    chosen=scores[target]
    above=sum(s>chosen+1e-12 for s in scores)
    tied=sum(abs(s-chosen)<=1e-12 for s in scores)
    hit=sum(pos<=k for pos in range(above+1,above+tied+1))/tied
    ndcg=sum(1/math.log2(pos+1) for pos in range(above+1,above+tied+1) if pos<=k)/tied
    return hit,ndcg


def evaluate(catalog, observations, algorithm, model=None, k=3):
    if not observations or k<1:
        raise ValueError("nonempty evaluation observations and k>=1 required")
    per_profile={}
    for o in observations:
        rows,target=design(catalog,o)
        scores=[score(algorithm,x,model,w) for x,w in zip(rows,shown_work_ids(catalog,o))]
        loss=-math.log(max(1e-15,probabilities(scores)[target]))
        hit,ndcg=(None,None) if target==len(rows) else tie_metrics(scores,target,k)
        per_profile.setdefault(o.request.profile_id,[]).append((loss,hit,ndcg))
    profile_values={}
    for profile,items in per_profile.items():
        positives=[x for x in items if x[1] is not None]
        profile_values[profile]={"choice_log_loss":sum(v[0] for v in items)/len(items),
            "hit_at_k":sum(v[1] for v in positives)/len(positives) if positives else None,
            "ndcg_at_k":sum(v[2] for v in positives)/len(positives) if positives else None}
    def average(key):
        values=[v[key] for v in profile_values.values() if v[key] is not None]
        return sum(values)/len(values) if values else None
    return {"algorithm":algorithm,"requests":len(observations),"profiles":len(per_profile),"k":k,
            "choice_log_loss":average("choice_log_loss"),"hit_at_k":average("hit_at_k"),
            "ndcg_at_k":average("ndcg_at_k"),"per_profile":profile_values,
            "metric_scope":"shown eligible choice sets; ranking conditional on a non-outside choice; tie-aware, profile-macro"}


def bootstrap_difference(candidate, baseline, metric="ndcg_at_k", seed=19, samples=1000):
    keys=sorted(set(candidate["per_profile"]) & set(baseline["per_profile"]))
    differences=[candidate["per_profile"][p][metric]-baseline["per_profile"][p][metric]
                 for p in keys if candidate["per_profile"][p][metric] is not None and baseline["per_profile"][p][metric] is not None]
    if not differences:
        return {"mean":None,"ci95":None,"profiles":0}
    rng=random.Random(seed)
    draws=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(samples))
    return {"mean":sum(differences)/len(differences),"ci95":[draws[int(.025*samples)],draws[min(samples-1,int(.975*samples))]],"profiles":len(differences)}


def compare(catalog, observations, validation_at, test_at, k=3, epochs=160):
    train,valid,test=temporal_split(observations,validation_at,test_at)
    choice=fit(catalog,train,"choice",epochs=epochs)
    pairwise=fit(catalog,train,"pairwise",epochs=epochs)
    # Residual model: hyperparameters chosen by validation choice log loss (a proper score).
    trials=[]
    for config in RESIDUAL_GRID:
        candidate=fit_residual(catalog,train,epochs=epochs,**config)
        trials.append((evaluate(catalog,valid,"residual",candidate,k)["choice_log_loss"],len(trials),candidate,config))
    residual_loss,_,residual,residual_config=min(trials,key=lambda t:t[:2])
    residual.metadata={**residual.metadata,"selected_config":residual_config,"selected_on":"validation choice_log_loss",
                       "grid_size":len(RESIDUAL_GRID)}
    # Select a blend and champion on validation loss, never on test results.
    blends=[Model("adaptive",choice.weights,a,{**choice.metadata,"blend_selected_on":"validation"}) for a in (0.,.25,.5,.75,1.)]
    adaptive=min(blends,key=lambda m:evaluate(catalog,valid,"adaptive",m,k)["choice_log_loss"])
    adaptive.metadata={**adaptive.metadata,"last_observed_at":max(valid,key=lambda o:instant(o.observed_at)).observed_at}
    models={"choice":choice,"pairwise":pairwise,"residual":residual,"adaptive":adaptive}
    names=[*WEIGHTS,*models]
    validation={n:evaluate(catalog,valid,n,models.get(n),k) for n in names}
    champion=min(names,key=lambda n:validation[n]["choice_log_loss"])
    # A ranking-specific selector can keep a simple baseline when it wins validation.
    # This adds selection budget (seven candidates), no extra model fitting or test access.
    rank_champion=max(names,key=lambda n:((validation[n]["ndcg_at_k"] if validation[n]["ndcg_at_k"] is not None else -1),-validation[n]["choice_log_loss"]))
    if rank_champion in WEIGHTS:
        selected_weights=list(WEIGHTS[rank_champion])
    elif rank_champion=="adaptive":
        selected_weights=[adaptive.blend*w+(1-adaptive.blend)*h for w,h in zip(adaptive.weights,WEIGHTS["hybrid"])]
    else:
        selected_weights=list(models[rank_champion].weights)
    selected_extra=residual.extra if rank_champion=="residual" else None
    models["selected"]=Model("selected",selected_weights,extra=selected_extra,metadata={**choice.metadata,
        "last_observed_at":max(valid,key=lambda o:instant(o.observed_at)).observed_at,
        "selected_from":rank_champion,"selection_metric":"validation profile-macro tie-aware NDCG@k",
        "selection_k":k,
        "validation_cutoff":test_at,"selection_candidates":list(names)})
    names.append("selected")
    validation["selected"]=evaluate(catalog,valid,"selected",models["selected"],k)
    results={n:evaluate(catalog,test,n,models.get(n),k) for n in names}
    best_base=min(WEIGHTS,key=lambda n:validation[n]["choice_log_loss"])
    intervals={n:bootstrap_difference(r,results[best_base]) for n,r in results.items()}
    clean=lambda r:{k:v for k,v in r.items() if k!="per_profile"}
    report={"synthetic_data":catalog.synthetic,"split":{"train":len(train),"validation":len(valid),"test":len(test),
        "validation_at":validation_at,"test_at":test_at,"profile_disjoint":True,"purged":len(observations)-sum(map(len,(train,valid,test)))},
        "selected_on_validation":champion,"ranking_selected_on_validation":rank_champion,"validation_selected_baseline":best_base,
        "validation":{n:clean(r) for n,r in validation.items()},"test":{n:clean(r) for n,r in results.items()},
        "ndcg_difference_vs_validation_baseline":intervals,
        "adaptive_blend":adaptive.blend,"residual_config":residual_config,
        "conclusion":"Controlled synthetic result, not evidence of real audience uplift." if catalog.synthetic else "Held-out association/choice evaluation; does not establish causal uplift or universal superiority."}
    return report,models
