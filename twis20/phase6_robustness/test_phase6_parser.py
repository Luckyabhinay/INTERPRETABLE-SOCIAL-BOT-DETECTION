"""Regression test: CSV NumPy representations must never be evaluated."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from twis20.phase6_robustness.run_phase6 import json_safe,load_candidate_records,validate_selected_record
class CandidateRecordTests(unittest.TestCase):
 def test_structured_records_ignore_numpy_like_csv_metrics(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); (root/'development_all_candidates.csv').write_text("metrics\n{'macro_f1': np.float64(0.8)}\n",encoding='utf-8')
   r={'experiment':'profile_text_network','algorithm':'xgboost','setting':'depth_8','status':'complete','feature_input_fingerprint':'fingerprint','metrics':{'macro_f1':.8,'pr_auc':.9},'duration_seconds':1,'model_file':'candidate_models/model.joblib'}
   (root/'candidate.json').write_text(json.dumps(r),encoding='utf-8')
   records=load_candidate_records(root,'fingerprint'); locked={'selection':{**{k:r[k] for k in ('experiment','algorithm','setting')},'metrics':r['metrics']},'selection_metadata':{'feature_input_fingerprint':'fingerprint'}}
   self.assertEqual(validate_selected_record(records,locked)['setting'],'depth_8')
 def test_json_safe_recursively_converts_numpy_values(self):
  payload={'integer':np.int64(7),'float':np.float64(.25),'boolean':np.bool_(True),'nested':{'array':np.array([np.int64(2),np.float64(3.5)])},'ordinary':[None,'safe']}
  converted=json_safe(payload)
  self.assertEqual(converted,{'integer':7,'float':.25,'boolean':True,'nested':{'array':[2,3.5]},'ordinary':[None,'safe']})
  json.dumps(converted)
if __name__=='__main__': unittest.main()
