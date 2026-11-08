"""Full-choice-set conditional logit and a pairwise logistic baseline.

Models learn only from explicitly supplied exposures/choices. No invented propensity,
no pickle loading, no outcome features and no random train/test split.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from .schema import Request, instant, number
from .engine import BASIS, FEATURES, LEARNED, WEIGHTS, basis, candidates, dot


def probabilities(scores):
    maximum = max([0., *scores])  # outside option has fixed utility zero
    values = [math.exp(s-maximum) for s in scores] + [math.exp(-maximum)]
    total = sum(values)
    return [v/total for v in values]


@dataclass(frozen=True)
class Observation:
    request: Request
    shown_ids: tuple[str, ...]
    chosen_id: str | None
    observed_at: str

    def __post_init__(self):
        if not self.shown_ids or len(set(self.shown_ids)) != len(self.shown_ids):
            raise ValueError("shown_ids must be nonempty and unique")
        if self.chosen_id is not None and self.chosen_id not in self.shown_ids:
            raise ValueError("choice must be shown or null for the outside option")
        if instant(self.observed_at) < instant(self.request.at):
            raise ValueError("feedback cannot precede request")
        object.__setattr__(self, "shown_ids", tuple(self.shown_ids))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(Request(**d["request"]), tuple(d["shown_ids"]), d["chosen_id"], d["observed_at"])


def shown_work_ids(catalog, observation):
    works = {s.production_id: s.work_id for s in catalog.shows}
    perf = {p.performance_id: works[p.production_id] for p in catalog.performances}
    return [perf[i] for i in observation.shown_ids]


def design(catalog, observation):
    eligible, _ = candidates(catalog, observation.request)
    vectors = {p.performance_id:x for _,p,x,_ in eligible}
    if any(i not in vectors for i in observation.shown_ids):
        raise ValueError("observation contains unknown, future, or infeasible shown item")
    rows = [vectors[i] for i in observation.shown_ids]
    target = observation.shown_ids.index(observation.chosen_id) if observation.chosen_id is not None else len(rows)
    return rows, target


@dataclass
class Model:
    method: str
    weights: list[float]
    blend: float = 1.
    metadata: dict | None = None
    # Optional residual terms: {"basis_weights": [...len(BASIS)], "work_bias": {work_id: float}}
    extra: dict | None = None

    def __post_init__(self):
        if self.method not in LEARNED or len(self.weights) != len(FEATURES):
            raise ValueError("invalid model method or feature dimension")
        for w in self.weights:
            number(w, "model weight")
        number(self.blend, "blend", 0, 1)
        if self.extra is not None:
            if self.method not in {"residual", "selected"} or set(self.extra) - {"basis_weights", "work_bias"}:
                raise ValueError("residual terms only allowed on residual/selected models")
            bw = self.extra.get("basis_weights")
            if bw is not None and len(bw) != len(BASIS):
                raise ValueError("basis dimension mismatch")
            for w in [*(bw or []), *self.extra.get("work_bias", {}).values()]:
                number(w, "residual weight")

    @property
    def model_id(self):
        body={"method":self.method,"weights":self.weights,"blend":self.blend,"metadata":self.metadata}
        if self.extra is not None:
            body["extra"]=self.extra
        return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()[:16]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload={"format_version":1,"features":list(FEATURES),**asdict(self),"model_id":self.model_id}
        if self.extra is None:
            payload.pop("extra")
        else:
            payload.update(format_version=2,basis=list(BASIS))
        Path(path).write_text(json.dumps(payload,allow_nan=False,indent=2)+"\n")

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d.get("format_version") not in (1, 2) or d.get("features") != list(FEATURES):
            raise ValueError("incompatible model format or feature order")
        if d["format_version"] == 2 and d.get("basis") != list(BASIS):
            raise ValueError("incompatible residual basis")
        result=cls(d["method"], d["weights"], d["blend"], d.get("metadata"), d.get("extra"))
        if result.model_id != d.get("model_id"):
            raise ValueError("model hash mismatch")
        return result


def fit(catalog, observations, method="choice", epochs=160, lr=.08, l2=.01):
    if method not in {"choice","pairwise"} or not observations or epochs < 1 or lr <= 0 or l2 < 0:
        raise ValueError("invalid training inputs")
    matrices=[design(catalog,o) for o in observations]
    weights=list(WEIGHTS["hybrid"])
    momentum=[0.]*len(weights)
    variance=[0.]*len(weights)
    for step in range(1,epochs+1):
        grad=[0.]*len(weights)
        for rows,target in matrices:
            extended=rows+[[0.]*len(weights)]
            if method=="choice":
                p=probabilities([dot(weights,x) for x in rows])
                for j,x in enumerate(extended):
                    error=p[j]-float(j==target)
                    for k,value in enumerate(x):
                        grad[k]+=error*value
            else:
                negatives=[x for j,x in enumerate(extended) if j!=target]
                chosen=extended[target]
                for negative in negatives:
                    delta=[a-b for a,b in zip(chosen,negative)]
                    margin=dot(weights,delta)
                    error=-1/(1+math.exp(max(-40,min(40,margin))))/len(negatives)
                    for k,value in enumerate(delta):
                        grad[k]+=error*value
        for k in range(len(weights)):
            g=grad[k]/len(matrices)+l2*weights[k]
            momentum[k]=.9*momentum[k]+.1*g
            variance[k]=.999*variance[k]+.001*g*g
            weights[k]-=lr*(momentum[k]/(1-.9**step))/(math.sqrt(variance[k]/(1-.999**step))+1e-8)
    data_hash=hashlib.sha256(json.dumps([o.to_dict() for o in observations],sort_keys=True).encode()).hexdigest()
    return Model(method,weights,metadata={"train_rows":len(observations),"epochs":epochs,"l2":l2,
        "last_observed_at":max(observations,key=lambda o:instant(o.observed_at)).observed_at,
        "train_sha256":data_hash,"synthetic_training":catalog.synthetic})


def fit_residual(catalog, observations, epochs=160, lr=.08, l2=.01, l2_basis=.01, l2_work=.05,
                 use_basis=True, use_work=True):
    """Conditional logit (with outside option) plus a hinge/cross basis and per-work effects.

    U_ij = w.x_ij + v.phi(x_ij) + b_work(j);  U_outside = 0.
    Penalised MLE: l2 pulls w toward the hybrid prior (MAP with a Gaussian prior), l2_basis
    shrinks v toward 0 (linear model as the fallback), l2_work shrinks unseen/rare works to 0.
    """
    if not observations or epochs < 1 or lr <= 0 or min(l2, l2_basis, l2_work) < 0:
        raise ValueError("invalid training inputs")
    data=[]
    for o in observations:
        rows,target=design(catalog,o)
        data.append((rows,[basis(x) for x in rows],shown_work_ids(catalog,o),target))
    prior=list(WEIGHTS["hybrid"])
    weights,vb=list(prior),[0.]*len(BASIS)
    keys=sorted({k for *_,works,_ in data for k in works}) if use_work else []
    bias={k:0. for k in keys}
    size=len(weights)+len(vb)+len(keys)
    momentum,variance=[0.]*size,[0.]*size
    for step in range(1,epochs+1):
        gw,gv,gb=[0.]*len(weights),[0.]*len(vb),{k:0. for k in keys}
        for rows,phis,works,target in data:
            p=probabilities([dot(weights,x)+dot(vb,f)+bias.get(k,0.) for x,f,k in zip(rows,phis,works)])
            for j,(x,f,k) in enumerate(zip(rows,phis,works)):
                error=p[j]-float(j==target)
                for i,value in enumerate(x):
                    gw[i]+=error*value
                if use_basis:
                    for i,value in enumerate(f):
                        gv[i]+=error*value
                if use_work:
                    gb[k]+=error
        n=len(data)
        grads=[g/n+l2*(w-a) for g,w,a in zip(gw,weights,prior)]
        grads+=[g/n+l2_basis*v for g,v in zip(gv,vb)]
        grads+=[gb[k]/n+l2_work*bias[k] for k in keys]
        for i,g in enumerate(grads):
            momentum[i]=.9*momentum[i]+.1*g
            variance[i]=.999*variance[i]+.001*g*g
            delta=lr*(momentum[i]/(1-.9**step))/(math.sqrt(variance[i]/(1-.999**step))+1e-8)
            if i<len(weights):
                weights[i]-=delta
            elif i<len(weights)+len(vb):
                vb[i-len(weights)]-=delta
            else:
                bias[keys[i-len(weights)-len(vb)]]-=delta
    data_hash=hashlib.sha256(json.dumps([o.to_dict() for o in observations],sort_keys=True).encode()).hexdigest()
    extra={"basis_weights":vb if use_basis else None,"work_bias":bias}
    return Model("residual",weights,extra=extra,metadata={"train_rows":len(observations),"epochs":epochs,
        "l2_prior":l2,"l2_basis":l2_basis if use_basis else None,"l2_work":l2_work if use_work else None,
        "last_observed_at":max(observations,key=lambda o:instant(o.observed_at)).observed_at,
        "train_sha256":data_hash,"synthetic_training":catalog.synthetic})


def temporal_split(observations, validation_at, test_at):
    val,test=instant(validation_at),instant(test_at)
    if val>=test:
        raise ValueError("validation cutoff must precede test cutoff")
    if len({o.request.request_id for o in observations}) != len(observations):
        raise ValueError("duplicate request IDs")
    # Keep the label window within a partition. Requests straddling a cutoff are purged.
    train=[o for o in observations if instant(o.observed_at)<val]
    valid=[o for o in observations if val<=instant(o.request.at) and instant(o.observed_at)<test]
    holdout=[o for o in observations if instant(o.request.at)>=test]
    future_profiles={o.request.profile_id for o in valid+holdout}
    test_profiles={o.request.profile_id for o in holdout}
    train=[o for o in train if o.request.profile_id not in future_profiles]
    valid=[o for o in valid if o.request.profile_id not in test_profiles]
    if not all((train,valid,holdout)):
        raise ValueError("split/purge leaves an empty partition")
    return train,valid,holdout
