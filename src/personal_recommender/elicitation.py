"""A transparent attribute-disagreement heuristic, not a validated dialogue policy."""
import math
from .engine import candidates


def suggest_question(catalog,request):
    rows,_=candidates(catalog,request)
    unique={s.work_id:s for s,_,_,_ in rows}
    if len(unique)<2:
        return None
    known=set(request.preferences)|set(request.avoid_tags)
    questions=[]
    for tag in sorted({t for s in unique.values() for t in s.tags}-known):
        p=sum(tag in s.tags for s in unique.values())/len(unique)
        if 0<p<1:
            entropy=-p*math.log2(p)-(1-p)*math.log2(1-p)
            questions.append({"attribute":tag,"question":f"Would you enjoy a production emphasizing {tag}?",
                              "options":["prefer","avoid","unsure"],"candidate_split_entropy":entropy,
                              "method":"attribute-disagreement heuristic; no outcome guarantee"})
    return max(questions,key=lambda q:q["candidate_split_entropy"]) if questions else None
