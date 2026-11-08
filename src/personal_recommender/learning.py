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
from .engine import FEATURES, WEIGHTS, candidates, dot


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

    def __post_init__(self):
        if self.method not in {"pairwise", "choice", "adaptive", "selected"} or len(self.weights) != len(FEATURES):
            raise ValueError("invalid model method or feature dimension")
        for w in self.weights:
            number(w, "model weight")
        number(self.blend, "blend", 0, 1)

    @property
    def model_id(self):
        blob=json.dumps({"method":self.method,"weights":self.weights,"blend":self.blend,"metadata":self.metadata},sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload={"format_version":1,"features":list(FEATURES),**asdict(self),"model_id":self.model_id}
        Path(path).write_text(json.dumps(payload,allow_nan=False,indent=2)+"\n")

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d.get("format_version") != 1 or d.get("features") != list(FEATURES):
            raise ValueError("incompatible model format or feature order")
        result=cls(d["method"], d["weights"], d["blend"], d.get("metadata"))
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
