"""Phase 6 robustness analysis of fixed Phase 4 candidates; no fitting occurs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,roc_auc_score,average_precision_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]; F=ROOT/'phase3_features_phase4'; P4=ROOT/'phase4_results_run03'; P5=ROOT/'phase5_explainability/outputs_run02'; MODEL=P4/'candidate_models/profile_text_network_xgboost_depth_8.joblib'; TH=.56
GROUPS=('profile','text_summary','text_tfidf','network'); EXP={'profile_only':('profile',),'text_only':('text_summary','text_tfidf'),'profile_text':('profile','text_summary','text_tfidf'),'profile_network':('profile','network'),'profile_text_network':GROUPS}
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def metrics(y,p):
 z=(p>=TH).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel(); return {'n':len(y),'genuine_n':int((y==0).sum()),'bot_n':int((y==1).sum()),'accuracy':accuracy_score(y,z),'precision_bot':precision_score(y,z,zero_division=0),'recall_bot':recall_score(y,z,zero_division=0),'macro_f1':f1_score(y,z,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,z),'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'false_positive_rate':fp/(fp+tn),'tn':tn,'fp':fp,'fn':fn,'tp':tp}
def load():
 man=json.loads((F/'feature_manifest.json').read_text(encoding='utf-8')); names=[n for g in GROUPS for n in man['feature_names'][g]]; b={g:sparse.load_npz(F/f'features_{g}.npz').tocsr() for g in GROUPS}; X=sparse.hstack([b[g] for g in GROUPS],format='csr',dtype=np.float32); meta=pd.read_json(F/'metadata.json',encoding='utf-8'); m=joblib.load(MODEL)
 if X.shape[1]!=len(names)!=0 or m.n_features_in_!=len(names):raise RuntimeError('model/manifest mismatch')
 return man,names,b,X,meta,m
def load_candidate_records(directory: Path, fingerprint: str) -> list[dict]:
 """Read only structured Phase 4 candidate JSON records; never parse CSV metrics."""
 records=[]
 for path in sorted(directory.glob('*.json')):
  record=json.loads(path.read_text(encoding='utf-8'))
  if record.get('status')!='complete': continue
  if record.get('feature_input_fingerprint')!=fingerprint: raise RuntimeError(f'Candidate fingerprint mismatch: {path.name}')
  if not isinstance(record.get('metrics'),dict) or 'macro_f1' not in record['metrics']: raise RuntimeError(f'Malformed candidate metrics: {path.name}')
  record['_record_path']=path; records.append(record)
 return records
def validate_selected_record(records: list[dict], locked: dict) -> dict:
 selection=locked['selection']; matches=[r for r in records if all(r.get(k)==selection[k] for k in ('experiment','algorithm','setting'))]
 if len(matches)!=1: raise RuntimeError('Locked selected candidate record is missing or ambiguous')
 record=matches[0]
 if record['metrics']!=selection['metrics'] or record['feature_input_fingerprint']!=locked['selection_metadata']['feature_input_fingerprint']: raise RuntimeError('Locked selection/candidate record mismatch')
 return record
def json_safe(value):
 """Recursively convert NumPy scalar/container values for JSON artifacts."""
 if isinstance(value,np.integer): return int(value)
 if isinstance(value,np.floating): return float(value)
 if isinstance(value,np.bool_): return bool(value)
 if isinstance(value,np.ndarray): return [json_safe(item) for item in value.tolist()]
 if isinstance(value,dict): return {key:json_safe(item) for key,item in value.items()}
 if isinstance(value,(list,tuple)): return [json_safe(item) for item in value]
 return value
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs_run01');a=ap.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=False);(o/'plots').mkdir()
 before={'model':sha(MODEL),'locked_selection':sha(P4/'locked_selection_before_test.json'),'test_metrics':sha(P4/'official_test_metrics.csv'),'phase5_report':sha(P5/'phase5_explainability_report.md')}; run=json.loads((P4/'run_manifest.json').read_text(encoding='utf-8')); feature_hash={n:sha(F/n) for n in run['feature_inputs']}
 if any(feature_hash[n]!=run['feature_inputs'][n]['sha256'] for n in feature_hash):raise RuntimeError('Phase 3 provenance hash mismatch')
 locked=json.loads((P4/'locked_selection_before_test.json').read_text(encoding='utf-8')); records=load_candidate_records(P4/'candidate_records',locked['selection_metadata']['feature_input_fingerprint']); selected_record=validate_selected_record(records,locked)
 man,names,blocks,X,meta,model=load(); test=np.flatnonzero(meta.split.eq('test')); y=meta.loc[test,'label'].astype(int).to_numpy(); prob=model.predict_proba(X[test])[:,1]; base=metrics(y,prob); pd.DataFrame([{'scenario':'unchanged_baseline',**base}]).to_csv(o/'baseline_metrics.csv',index=False)
 # A: overlap cohorts, descriptive only; no membership changes.
 rows=[]
 for flag in ('text_overlap_exact_train','text_overlap_normalized_train'):
  for val,label in ((True,'overlap'),(False,'non_overlap')):
   mask=meta.loc[test,flag].fillna(False).astype(bool).to_numpy()==val; rows.append({'flag':flag,'cohort':label,**metrics(y[mask],prob[mask])})
 pd.DataFrame(rows).to_csv(o/'overlap_sensitivity.csv',index=False)
 # B: fixed-candidate comparison. Best stored dev candidate per experiment; no optimization now.
 out=[]
 for exp in EXP:
   choices=[r for r in records if r['experiment']==exp]; chosen=sorted(choices,key=lambda r:(-r['metrics']['macro_f1'],-r['metrics']['pr_auc'],r['duration_seconds']))[0]; cand=joblib.load(P4/chosen['model_file']); XX=sparse.hstack([blocks[g] for g in EXP[exp]],format='csr',dtype=np.float32); pp=cand.predict_proba(XX[test])[:,1]; out.append({'experiment':exp,'model_file':chosen['model_file'],'dev_macro_f1':chosen['metrics']['macro_f1'],'threshold':chosen['threshold'],**metrics(y,pp)})
 pd.DataFrame(out).to_csv(o/'feature_ablation_fixed_models.csv',index=False)
 # C/D fixed-model stress tests. Replace only profile numeric values with neutral zero (a documented stress test); unavailable scenarios are recorded rather than invented.
 perturb=[]; profile_idx=np.arange(len(man['feature_names']['profile'])); numeric=[i for i,n in enumerate(man['feature_names']['profile']) if n.startswith('profile.') and not n.endswith('__missing') and '__true' not in n and '__false' not in n and '__unknown' not in n]
 for name,fn,kind in [('profile_numeric_zero_stress',lambda z: z.__setitem__((slice(None),numeric),0),'stress_test'),('profile_numeric_plus_5pct',lambda z: z.__setitem__((slice(None),numeric),z[:,numeric]*1.05),'reasonable_noise')]:
  xb=X[test].tolil(copy=True); fn(xb); pp=model.predict_proba(xb.tocsr())[:,1]; perturb.append({'scenario':name,'scenario_type':kind,'simulation':'Only already-imputed profile numeric columns changed; original matrices untouched. Zero scenario is stress-only, not natural missingness.','delta_macro_f1':metrics(y,pp)['macro_f1']-base['macro_f1'],**metrics(y,pp)})
 for name in ('missing_profile_information','missing_tweet_summaries','reduced_tweet_availability','missing_network_information'):
  perturb.append({'scenario':name,'scenario_type':'unavailable','simulation':'Not run: valid simulation requires raw-field reconstruction through the original preprocessor or a defined missingness indicator; no arbitrary feature replacement applied.'})
 pd.DataFrame(perturb).to_csv(o/'missingness_and_noise_robustness.csv',index=False)
 # E/F: errors and legitimate groups only. Demographics unavailable; no inference from identity text.
 pred=(prob>=TH).astype(int); er=pd.DataFrame({'error_type':np.where((y==0)&(pred==1),'false_positive',np.where((y==1)&(pred==0),'false_negative','correct'))}); er.groupby('error_type').size().rename('count').reset_index().to_csv(o/'error_counts.csv',index=False)
 fairness=pd.DataFrame([{'audit':'demographic_fairness','status':'unavailable','reason':'No protected or demographic subgroup labels are present; none were inferred from account identity or text.'},{'audit':'legitimate_non_sensitive_groups','status':'unavailable','reason':'No approved non-sensitive subgroup definition was supplied; overlap cohorts are reported separately as robustness diagnostics.'}]);fairness.to_csv(o/'fairness_audit.csv',index=False)
 after={'model':sha(MODEL),'locked_selection':sha(P4/'locked_selection_before_test.json'),'test_metrics':sha(P4/'official_test_metrics.csv'),'phase5_report':sha(P5/'phase5_explainability_report.md')};
 if before!=after:raise RuntimeError('Immutable artifact changed during Phase 6')
 val={'hashes_before':before,'hashes_after':after,'phase3_hashes':feature_hash,'model_inputs':int(model.n_features_in_),'feature_names':len(names),'baseline':base,'test_label_tuning':False,'selected_candidate_identity':{k:selected_record[k] for k in ('experiment','algorithm','setting')},'phase5_outputs_present':all((P5/x).exists() for x in ['validation.json','global_shap_importance.csv','phase5_explainability_report.md'])};(o/'validation.json').write_text(json.dumps(json_safe(val),indent=2),encoding='utf-8')
 report=f'''# Phase 6 — Robustness, Fairness and Generalization\n\n## Integrity\nThe fixed Phase 4 XGBoost model, locked selection, official metrics, Phase 3 inputs and Phase 5 report had identical hashes before/after. No model fitting occurred and threshold remained {TH}.\n\n## Overlap\n`overlap_sensitivity.csv` reports all official-test overlap cohorts at the locked threshold. These are exposure diagnostics: copied/public text can affect benchmark performance but does not prove account-level leakage or memorization.\n\n## Fixed-model comparison\n`feature_ablation_fixed_models.csv` compares best stored Phase 4 candidates by feature configuration. This is a model comparison, not causal ablation; true removal needs retraining and is deferred.\n\n## Missingness/noise\nOnly documented profile numeric stress/noise scenarios were executed on copies. Raw-text/profile missingness and network removal were unavailable without valid raw reconstruction and are not fabricated.\n\n## Fairness and generalization\nDemographic fairness cannot be established: no demographic/protected labels are available, and none were inferred. Generalization is limited by text dependence, verification contribution, text overlap sensitivity, low network contribution in Phase 5, transductive topology, and TwiBot-20 representativeness. Correlations and SHAP attributions are not causal claims.\n''';(o/'phase6_robustness_report.md').write_text(report,encoding='utf-8')
if __name__=='__main__':main()
