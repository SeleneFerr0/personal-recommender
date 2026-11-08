from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from personal_recommender.schema import Catalog,Performance,Request,Show,instant
from personal_recommender.engine import candidates,eligibility,features,recommend,FEATURES
from personal_recommender.learning import Model,Observation,fit,probabilities,temporal_split,design
from personal_recommender.evaluation import tie_metrics,evaluate,compare
from personal_recommender.synthetic import demo_request,generate,make_catalog
from personal_recommender.elicitation import suggest_question
from personal_recommender.cli import main


class Constraints(unittest.TestCase):
    def setUp(self):
        self.cat=make_catalog()
        self.req=demo_request()
        self.show=self.cat.shows[0]
        self.perf=self.cat.performances[0]

    def reasons(self,perf=None,req=None,show=None):
        return eligibility(show or self.show,perf or self.perf,req or self.req)

    def test_total_budget(self):
        p=replace(self.perf,price=100)
        self.assertIn("over_budget",self.reasons(p,replace(self.req,budget=150))[0])
        self.assertNotIn("over_budget",self.reasons(p,replace(self.req,budget=200))[0])

    def test_seats_and_party(self):
        self.assertIn("insufficient_seats",self.reasons(replace(self.perf,available_seats=1,consecutive_seats=1))[0])
        self.assertIn("insufficient_consecutive_seats",self.reasons(replace(self.perf,consecutive_seats=1))[0])

    def test_unknown_seats_strict_and_permissive(self):
        p=replace(self.perf,available_seats=None,consecutive_seats=None)
        self.assertIn("requires_verification",self.reasons(p)[0])
        cat=Catalog([self.show],[p],self.cat.snapshot_at,True)
        out=recommend(cat,replace(self.req,allow_unverified=True))
        self.assertEqual(out["results"][0]["bookability"],"requires_verification")

    def test_stale_unknown_not_affordable_claim(self):
        p=replace(self.perf,expires_at="2030-01-02T00:00:00Z",price=1)
        self.assertIn("requires_verification",self.reasons(p)[0])
        self.assertEqual(features(self.show,p,replace(self.req,allow_unverified=True))[3],0)

    def test_future_snapshot_excluded_even_permissive(self):
        p=replace(self.perf,verified_at="2030-01-04T00:00:00Z")
        self.assertIn("future_ticket_snapshot",self.reasons(p,replace(self.req,allow_unverified=True))[0])

    def test_known_at_and_catalog_asof(self):
        p=replace(self.perf,known_at="2030-01-04T00:00:00Z",verified_at="2030-01-04T00:00:00Z")
        self.assertIn("not_known_at_request",self.reasons(p)[0])
        with self.assertRaisesRegex(ValueError,"catalog snapshot"):
            recommend(replace(self.cat,snapshot_at="2030-01-04T00:00:00Z"),self.req)

    def test_cancelled_and_soldout(self):
        for status in ("cancelled","sold_out"):
            self.assertIn(status,self.reasons(replace(self.perf,status=status),replace(self.req,allow_unverified=True))[0])

    def test_currency_no_implicit_conversion(self):
        self.assertIn("price_or_currency",self.reasons(replace(self.perf,currency="GBP"))[1])

    def test_age_and_accessibility(self):
        self.assertIn("minimum_age",self.reasons(req=replace(self.req,youngest_age=5),show=replace(self.show,min_age=6))[0])
        self.assertIn("requires_verification",self.reasons(req=replace(self.req,youngest_age=5))[0])
        self.assertIn("accessibility_mismatch",self.reasons(replace(self.perf,accessible=False),replace(self.req,requires_accessibility=True))[0])

    def test_local_date_not_utc_date(self):
        p=replace(self.perf,starts_at="2030-01-10T23:30:00-05:00")
        r=replace(self.req,start_date="2030-01-10",end_date="2030-01-10")
        self.assertFalse(self.reasons(p,r)[0])

    def test_nonexistent_dst_wall_time_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.perf,starts_at="2030-03-10T02:30:00-05:00")

    def test_wrong_offset_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.perf,starts_at="2030-01-10T14:00:00+00:00")

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            instant("2030-01-03T12:00:00")

    def test_repeat_policy(self):
        r=replace(self.req,seen_work_ids=(self.show.work_id,),allow_repeats=False)
        self.assertIn("repeat_not_allowed",self.reasons(req=r)[0])

    def test_duplicate_and_unknown_references(self):
        with self.assertRaises(ValueError):
            Catalog([self.show,self.show],[],self.cat.snapshot_at)
        with self.assertRaises(ValueError):
            Catalog([], [self.perf],self.cat.snapshot_at)

    def test_invalid_numeric_and_boolean_inputs(self):
        for value in (float("nan"),float("inf"),True,-1):
            with self.assertRaises(ValueError):
                replace(self.req,budget=value)
        with self.assertRaises(ValueError):
            replace(self.req,party_size=True)
        with self.assertRaises(ValueError):
            replace(self.req,allow_unverified="yes")

    def test_group_aversion_not_dropped(self):
        r=replace(self.req,member_preferences=({self.show.tags[0]:-1},))
        self.assertLess(features(self.show,self.perf,r)[7],0)

    def test_dedup_by_work_not_performance(self):
        out=recommend(self.cat,self.req,k=12)
        ids=[r["work_id"] for r in out["results"]]
        self.assertEqual(len(ids),len(set(ids)))
        for row in out["results"]:
            self.assertAlmostEqual(row["score"],sum(row["score_contributions"].values()))

    def test_empty_candidates_and_unknown_algorithm(self):
        self.assertEqual(recommend(self.cat,replace(self.req,cities=("Tokyo",)))["results"],[])
        with self.assertRaises(ValueError):
            recommend(self.cat,self.req,"bad")

    def test_question_uses_eligible_unknown_attribute(self):
        q=suggest_question(self.cat,self.req)
        self.assertNotIn(q["attribute"],self.req.preferences)
        self.assertIsNone(suggest_question(self.cat,replace(self.req,cities=("Tokyo",))))

    def test_catalog_round_trip(self):
        self.assertEqual(Catalog.from_dict(json.loads(json.dumps(self.cat.to_dict()))).to_dict(),self.cat.to_dict())


class Learning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat,cls.obs,cls.val,cls.test=generate(11,"mixed",60)

    def test_stable_probability_and_outside(self):
        p=probabilities([10000,-10000])
        self.assertAlmostEqual(sum(p),1)
        self.assertEqual(len(p),3)

    def test_tie_aware_ranking(self):
        hit,ndcg=tie_metrics([1,1,1],0,1)
        self.assertAlmostEqual(hit,1/3)
        self.assertAlmostEqual(ndcg,1/3)
        self.assertEqual(tie_metrics([3,2,1],0,1),(1,1))

    def test_label_validation(self):
        o=self.obs[0]
        with self.assertRaises(ValueError):
            replace(o,chosen_id="unshown")
        with self.assertRaises(ValueError):
            replace(o,observed_at="2029-01-01T00:00:00Z")
        with self.assertRaises(ValueError):
            replace(o,shown_ids=(o.shown_ids[0],o.shown_ids[0]))

    def test_observation_roundtrip(self):
        o=self.obs[0]
        self.assertEqual(Observation.from_dict(o.to_dict()).to_dict(),o.to_dict())

    def test_split_group_purge(self):
        obs=list(self.obs)
        obs[0]=replace(obs[0],request=replace(obs[0].request,profile_id=obs[-1].request.profile_id))
        train,val,test=temporal_split(obs,self.val,self.test)
        self.assertEqual(len(train),29)
        self.assertFalse({o.request.profile_id for o in train}&{o.request.profile_id for o in val+test})

    def test_label_window_purge(self):
        obs=list(self.obs)
        obs[0]=replace(obs[0],observed_at=self.val)
        train,val,test=temporal_split(obs,self.val,self.test)
        self.assertNotIn(obs[0],train+val+test)

    def test_duplicate_request_rejected(self):
        with self.assertRaises(ValueError):
            temporal_split([*self.obs,self.obs[0]],self.val,self.test)

    def test_infeasible_shown_item_rejected(self):
        o=self.obs[0]
        with self.assertRaises(ValueError):
            design(self.cat,replace(o,request=replace(o.request,cities=("Tokyo",))))

    def test_training_is_deterministic_and_reduces_train_loss(self):
        train,_,_=temporal_split(self.obs,self.val,self.test)
        a=fit(self.cat,train,epochs=40)
        b=fit(self.cat,train,epochs=40)
        self.assertEqual(a.model_id,b.model_id)
        self.assertLess(evaluate(self.cat,train,"choice",a)["choice_log_loss"],evaluate(self.cat,train,"hybrid")["choice_log_loss"])

    def test_model_roundtrip_corruption_and_feature_order(self):
        m=Model("choice",[0.]*len(FEATURES))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"model.json"
            m.save(p)
            self.assertEqual(Model.load(p).model_id,m.model_id)
            payload=json.loads(p.read_text());payload["weights"][0]=1;p.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                Model.load(p)
            m.save(p);payload=json.loads(p.read_text());payload["features"].reverse();p.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                Model.load(p)

    def test_future_model_rejected(self):
        m=Model("choice",[0.]*len(FEATURES),metadata={"last_observed_at":"2030-01-08T00:00:00Z"})
        with self.assertRaises(ValueError):
            recommend(self.cat,self.obs[0].request,"choice",m)

    def test_synthetic_model_rejected_on_real_catalog(self):
        m=Model("choice",[0.]*len(FEATURES),metadata={"synthetic_training":True})
        with self.assertRaisesRegex(ValueError,"synthetic-trained"):
            recommend(replace(self.cat,synthetic=False),self.obs[0].request,"choice",m)

    def test_outside_only_has_no_conditional_ranking_metric(self):
        o=replace(self.obs[0],chosen_id=None)
        result=evaluate(self.cat,[o],"hybrid")
        self.assertIsNone(result["ndcg_at_k"])
        self.assertGreater(result["choice_log_loss"],0)

    def test_full_compare_and_validation_selection(self):
        report,models=compare(self.cat,self.obs,self.val,self.test,epochs=12)
        self.assertTrue(report["split"]["profile_disjoint"])
        self.assertIn(report["selected_on_validation"],report["test"])
        self.assertTrue(report["synthetic_data"])
        self.assertIn(models["adaptive"].blend,(0,.25,.5,.75,1))
        winner=report["ranking_selected_on_validation"]
        self.assertAlmostEqual(report["validation"]["selected"]["ndcg_at_k"],report["validation"][winner]["ndcg_at_k"])

    def test_cli_demo(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"demo.json"
            self.assertEqual(main(["demo","--output",str(p)]),0)
            self.assertTrue(json.loads(p.read_text())["synthetic_data"])

    def test_selector_availability_includes_validation_labels(self):
        report,models=compare(self.cat,self.obs,self.val,self.test,epochs=5)
        with self.assertRaisesRegex(ValueError,"model trained"):
            recommend(self.cat,self.obs[0].request,"selected",models["selected"])

    def test_simulate_and_compare_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(main(["simulate","--output-dir",d,"--n","30"]),0)
            manifest=json.loads((root/"manifest.json").read_text())
            output=root/"report.json"
            self.assertEqual(main(["compare","--catalog",str(root/"catalog.json"),"--observations",str(root/"observations.json"),
                "--validation-at",manifest["validation_at"],"--test-at",manifest["test_at"],"--epochs","3","--output",str(output),
                "--model-dir",str(root/"models")]),0)
            self.assertIn("selected",json.loads(output.read_text())["test"])
            self.assertTrue((root/"models/selected.json").exists())


if __name__=="__main__":
    unittest.main()
