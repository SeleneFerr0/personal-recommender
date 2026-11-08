"""JSON-in/JSON-out command line interface."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import platform
from .schema import Catalog, Request
from .engine import recommend,WEIGHTS
from .learning import Model,Observation
from .evaluation import compare
from .synthetic import REGIMES, demo_request, generate,make_catalog
from .elicitation import suggest_question


def read(path):
    return json.loads(Path(path).read_text())


def write(path,data):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Personal theatre recommendation: fictional demo or supplied catalog/choices")
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo")
    demo.add_argument("--output")
    demo.add_argument("--export-dir")
    sim=sub.add_parser("simulate")
    sim.add_argument("--output-dir",required=True)
    sim.add_argument("--n",type=int,default=360)
    sim.add_argument("--seed",type=int,default=7)
    sim.add_argument("--regime",choices=REGIMES,default="mixed")
    rec=sub.add_parser("recommend")
    rec.add_argument("--catalog",required=True)
    rec.add_argument("--request",required=True)
    rec.add_argument("--algorithm",choices=[*WEIGHTS,"choice","pairwise","adaptive","selected"],default="hybrid")
    rec.add_argument("--model")
    rec.add_argument("--k",type=int,default=5)
    rec.add_argument("--output")
    train=sub.add_parser("compare")
    train.add_argument("--catalog",required=True)
    train.add_argument("--observations",required=True)
    train.add_argument("--validation-at",required=True)
    train.add_argument("--test-at",required=True)
    train.add_argument("--epochs",type=int,default=160)
    train.add_argument("--output",required=True)
    train.add_argument("--model-dir")
    bench=sub.add_parser("benchmark")
    bench.add_argument("--seeds",default="7,19,43")
    bench.add_argument("--regimes",default=",".join(REGIMES))
    bench.add_argument("--n",type=int,default=360)
    bench.add_argument("--epochs",type=int,default=120)
    bench.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="demo":
            catalog,request=make_catalog(),demo_request()
            result=recommend(catalog,request)
            result["next_question"]=suggest_question(catalog,request)
            if args.export_dir:
                write(Path(args.export_dir)/"catalog.json",catalog.to_dict())
                from dataclasses import asdict
                write(Path(args.export_dir)/"request.json",asdict(request))
        elif args.command=="simulate":
            catalog,rows,val,test=generate(args.seed,args.regime,args.n)
            write(Path(args.output_dir)/"catalog.json",catalog.to_dict())
            write(Path(args.output_dir)/"observations.json",[o.to_dict() for o in rows])
            result={"synthetic_data":True,"validation_at":val,"test_at":test,"n":args.n,"seed":args.seed,"regime":args.regime,
                    "warning":"Fictional productions, ticket snapshots and simulated choices; not real-user data."}
            write(Path(args.output_dir)/"manifest.json",result)
        elif args.command=="recommend":
            result=recommend(Catalog.from_dict(read(args.catalog)),Request(**read(args.request)),args.algorithm,
                             Model.load(args.model) if args.model else None,args.k)
        elif args.command=="compare":
            catalog=Catalog.from_dict(read(args.catalog))
            rows=[Observation.from_dict(d) for d in read(args.observations)]
            result,models=compare(catalog,rows,args.validation_at,args.test_at,epochs=args.epochs)
            result["input_sha256"]={"catalog":hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest(),
                                    "observations":hashlib.sha256(Path(args.observations).read_bytes()).hexdigest()}
            if args.model_dir:
                for name,model in models.items():
                    model.save(Path(args.model_dir)/f"{name}.json")
        else:
            start=time.perf_counter()
            runs=[]
            for seed in map(int,args.seeds.split(",")):
                for regime in args.regimes.split(","):
                    cat,rows,val,test=generate(seed,regime,args.n)
                    report,_=compare(cat,rows,val,test,epochs=args.epochs)
                    report.update(seed=seed,regime=regime)
                    runs.append(report)
                    print(f"completed seed={seed} regime={regime}",file=sys.stderr,flush=True)
            names=list(runs[0]["test"])
            aggregate={name:{metric:sum(r["test"][name][metric] for r in runs)/len(runs)
                             for metric in ("ndcg_at_k","hit_at_k","choice_log_loss")} for name in names}
            source_digest=hashlib.sha256()
            for file in sorted(Path(__file__).parent.glob("*.py")):
                source_digest.update(file.name.encode());source_digest.update(file.read_bytes())
            result={"synthetic_data":True,"warning":"Simulator shares interpretable features with trained models. No real-user validation, novelty claim, causal conclusion or universal superiority.",
                    "runs":runs,"aggregate":aggregate,"seconds":time.perf_counter()-start,
                    "implementation_sha256":source_digest.hexdigest(),"python":platform.python_version(),"platform":platform.platform(),
                    "config":{"n":args.n,"epochs":args.epochs,"seeds":args.seeds,"regimes":args.regimes}}
        if getattr(args,"output",None):
            write(args.output,result)
            print(f"Saved {args.output}")
        else:
            print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError) as exc:
        print(f"error: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
