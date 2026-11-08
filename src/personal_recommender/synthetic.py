"""Fictional data for executable demos and controlled stress tests, never Disney records."""
from datetime import datetime, timedelta, timezone
import random
from .schema import Catalog, Show, Performance, Request
from .engine import candidates, dot
from .learning import Observation, probabilities

TAGS=("adventure","comedy","dance","orchestra","family","romance","spectacle","drama")
# "latent": mixed utility plus a per-work appeal hidden from the features (reviews, word of
# mouth, IP strength). Tests whether a model can learn work-level effects from choice logs.
REGIMES=("content","context","popularity","mixed","nonlinear","latent")


def make_catalog(seed=7, size=16):
    rng=random.Random(seed)
    shows=[]
    performances=[]
    for i in range(size):
        shows.append(Show(f"work-{i}",f"production-{i}",f"Fictional musical {i+1}",tuple(rng.sample(TAGS,3)),
                          popularity=rng.random(),source="synthetic fixture; not a real production"))
        for hour in (14,19):
            performances.append(Performance(f"performance-{i}-{hour}",f"production-{i}","New York","Fictional venue",
                f"2030-01-{10+i%3:02d}T{hour:02d}:00:00-05:00","America/New_York","2030-01-01T00:00:00Z",
                price=float(rng.randrange(40,180)),available_seats=12,consecutive_seats=6,
                verified_at="2030-01-01T00:00:00Z",expires_at="2030-01-09T00:00:00Z",
                status="available",accessible=True,source="synthetic ticket fixture; never bookable"))
    return Catalog(shows,performances,"2030-01-01T00:00:00Z",synthetic=True)


def demo_request():
    return Request("demo","demo-profile","2030-01-03T12:00:00Z",("New York",),"2030-01-10","2030-01-12",300,
                   party_size=2,budget_mode="total",preferences={"family":1,"spectacle":.8,"drama":-.5},
                   member_preferences=({"family":1},{"dance":1,"comedy":.5}),require_consecutive=True)


def generate(seed=7, regime="mixed", n=360):
    if regime not in REGIMES or not 30<=n<=900:
        raise ValueError("unknown regime or n outside [30, 900] fixture time window")
    rng=random.Random(seed+101)
    catalog=make_catalog(seed)
    vectors={
        "content":[-.8,5,3,0,0,0,0,0,0,0,0,0],
        "context":[-.8,0,0,4,4,0,0,0,0,0,0,0],
        "popularity":[-1.,0,0,0,0,0,5,0,0,0,0,0],
        "mixed":[-1.5,2,3,1.5,1,.3,.4,2,1,3,2,0],
        "nonlinear":[-1.5,1,2,1,1,0,0,1,1,0,0,0],
        "latent":[-1.5,2,3,1.5,1,.3,.4,2,1,3,2,0],
    }
    appeal_rng=random.Random(seed+202)
    appeal={s.work_id:appeal_rng.gauss(0,1) for s in catalog.shows}
    observations=[]
    for i in range(n):
        prefs={tag:rng.uniform(.2,1) for tag in rng.sample(TAGS,3)}
        other=[t for t in TAGS if t not in prefs]
        prefs[rng.choice(other)]=-rng.uniform(.2,1)
        at=datetime(2030,1,2,tzinfo=timezone.utc)+timedelta(minutes=i*10)
        request=Request(f"sim-{seed}-{regime}-{i}",f"profile-{seed}-{i}",at.isoformat(),("New York",),
            "2030-01-10","2030-01-12",200,party_size=2,preferences=prefs,
            member_preferences=(prefs,{t:1. for t in rng.sample(TAGS,2)}),preferred_hour=rng.choice((14,18,20)))
        eligible,_=candidates(catalog,request)
        # Sample the displayed set independently of latent utility and deduplicate works.
        rng.shuffle(eligible)
        seen=set()
        shown=[]
        for row in eligible:
            if row[0].work_id not in seen:
                shown.append(row)
                seen.add(row[0].work_id)
            if len(shown)==8:
                break
        utilities=[dot(vectors[regime],x) for _,_,x,_ in shown]
        if regime=="nonlinear":
            import math
            utilities=[v+2*math.sin(7*x[1]*x[4])-3*(x[3]-.5)**2 for v,(_,_,x,_) in zip(utilities,shown)]
        if regime=="latent":
            utilities=[v+appeal[s.work_id] for v,(s,_,_,_) in zip(utilities,shown)]
        probs=probabilities(utilities)
        chosen=rng.choices(list(range(len(shown)+1)),weights=probs)[0]
        ids=tuple(p.performance_id for _,p,_,_ in shown)
        observations.append(Observation(request,ids,ids[chosen] if chosen<len(ids) else None,(at+timedelta(minutes=1)).isoformat()))
    val=(datetime(2030,1,2,tzinfo=timezone.utc)+timedelta(minutes=int(n*.5)*10)).isoformat()
    test=(datetime(2030,1,2,tzinfo=timezone.utc)+timedelta(minutes=int(n*.75)*10)).isoformat()
    return catalog,observations,val,test
