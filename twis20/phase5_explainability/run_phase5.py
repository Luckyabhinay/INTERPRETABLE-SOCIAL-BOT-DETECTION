"""Phase 5 SHAP explanations for the locked TwiBot-20 Phase 4 XGBoost model.

This is an explanation-only workflow: it never fits or mutates the model and
creates a fresh output directory rather than writing into Phase 3/4 artifacts.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import shap

ROOT=Path(__file__).resolve().parents[1]
FEATURES=ROOT/'phase3_features_phase4'; PHASE4=ROOT/'phase4_results_run03'
MODEL=PHASE4/'candidate_models/profile_text_network_xgboost_depth_8.joblib'
FILES=['feature_manifest.json','features_profile.npz','features_text_summary.npz','features_text_tfidf.npz','features_network.npz']
GROUPS=('profile','text_summary','text_tfidf','network'); SEED=20261106

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def fingerprints():
    return {
        **{f:sha(FEATURES/f) for f in FILES},
        'model':sha(MODEL),
        'selection_lock':sha(PHASE4/'locked_selection_before_test.json'),
        'official_test_metrics':sha(PHASE4/'official_test_metrics.csv'),
    }
def load_inputs():
    manifest=json.loads((FEATURES/'feature_manifest.json').read_text(encoding='utf-8'))
    names=[n for g in GROUPS for n in manifest['feature_names'][g]]
    X=sparse.hstack([sparse.load_npz(FEATURES/f'features_{g}.npz') for g in GROUPS],format='csr',dtype=np.float32)
    metadata=pd.read_json(FEATURES/'metadata.json',encoding='utf-8')
    model=joblib.load(MODEL)
    if X.shape[1]!=len(names) or len(names)!=len(set(names)) or getattr(model,'n_features_in_',None)!=len(names): raise RuntimeError('Feature mapping/model input mismatch')
    if model.predict_proba(X[:1]).shape!=(1,2): raise RuntimeError('Model rejected reconstructed CSR matrix')
    return model,X,metadata,manifest,names
def sample_indices(meta,n):
    dev=np.flatnonzero(meta['split'].eq('dev')); labels=meta.loc[dev,'label'].astype(int).to_numpy()
    n=min(n,len(dev)); split=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=SEED)
    picked,_=next(split.split(np.zeros(len(dev)),labels)); return dev[picked]
def feature_groups(names):
    out=[]
    for name in names:
        out.append('profile' if name.startswith('profile.') else 'tweet_summary' if name.startswith('tweet.') and not name.startswith('tweet.tfidf.') else 'tfidf' if name.startswith('tweet.tfidf.') else 'network')
    return np.asarray(out)
def markdown_table(frame):
    headers=list(frame.columns)
    rows=[[str(value) for value in row] for row in frame.itertuples(index=False,name=None)]
    widths=[max(len(str(header)),*(len(row[i]) for row in rows)) for i,header in enumerate(headers)]
    lines=['| '+' | '.join(str(header).ljust(widths[i]) for i,header in enumerate(headers))+' |']
    lines.append('| '+' | '.join('-'*width for width in widths)+' |')
    lines.extend('| '+' | '.join(row[i].ljust(widths[i]) for i in range(len(headers)))+' |' for row in rows)
    return '\n'.join(lines)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--sample-size',type=int,default=200); a=ap.parse_args()
    if a.sample_size<1: raise ValueError('--sample-size must be positive')
    out=a.output; out.mkdir(parents=True,exist_ok=False); plots=out/'plots'; plots.mkdir(); water=plots/'shap_waterfall_examples'; water.mkdir()
    before=fingerprints(); metrics_before=before['official_test_metrics']; model,X,meta,manifest,names=load_inputs(); recorded=json.loads((PHASE4/'run_manifest.json').read_text(encoding='utf-8'))
    lock=json.loads((PHASE4/'locked_selection_before_test.json').read_text(encoding='utf-8'))
    selection=lock.get('selection',{}); selection_metadata=lock.get('selection_metadata',{})
    if selection_metadata.get('selection_split')!='dev' or selection_metadata.get('test_metrics_used') is not False: raise RuntimeError('Phase 4 selection lock is not development-only')
    if (selection.get('experiment'),selection.get('algorithm'),selection.get('setting'))!=('profile_text_network','xgboost','depth_8'): raise RuntimeError('Locked candidate does not match the Phase 5 model')
    threshold=float(selection['threshold'])
    if any(before[f]!=recorded['feature_inputs'][f]['sha256'] for f in FILES): raise RuntimeError('Phase 3 fingerprint differs from Phase 4 provenance')
    indices=sample_indices(meta,a.sample_size); Xs=X[indices]; probs=model.predict_proba(Xs)[:,1]
    explainer=shap.TreeExplainer(model,model_output='raw'); values=np.asarray(explainer.shap_values(Xs),dtype=np.float64)
    if values.shape!=(len(indices),len(names)) or not np.isfinite(values).all(): raise RuntimeError('Invalid SHAP output')
    expected=np.asarray(model.get_booster().predict(__import__('xgboost').DMatrix(Xs),output_margin=True)); base=float(np.asarray(explainer.expected_value).reshape(-1)[0]); residual=np.abs(values.sum(axis=1)+base-expected)
    if residual.max()>1e-4: raise RuntimeError(f'SHAP additivity failed: {residual.max()}')
    importance=np.abs(values).mean(axis=0); group=feature_groups(names)
    imp=pd.DataFrame({'feature':names,'mean_abs_shap':importance,'feature_group':group}).sort_values('mean_abs_shap',ascending=False); imp.to_csv(out/'global_shap_importance.csv',index=False); imp.head(20).to_csv(out/'top20_features.csv',index=False); imp.head(30).to_csv(out/'top30_features.csv',index=False)
    group_df=imp.groupby('feature_group',as_index=False)['mean_abs_shap'].sum().sort_values('mean_abs_shap',ascending=False); group_df.to_csv(out/'feature_group_importance.csv',index=False)
    # Only the explanation sample is densified for SHAP plots; full CSR X is retained.
    dense=Xs.toarray(); shap.summary_plot(values,dense,feature_names=names,max_display=30,show=False); plt.tight_layout(); plt.savefig(plots/'shap_summary.png',dpi=160); plt.close()
    shap.summary_plot(values,dense,feature_names=names,plot_type='bar',max_display=30,show=False); plt.tight_layout(); plt.savefig(plots/'shap_bar.png',dpi=160); plt.close()
    # Select two predicted bot and two predicted genuine dev examples without outputting account IDs.
    predicted=(probs>=threshold).astype(int); examples=[]
    for wanted in (0,1):
        for pos in np.flatnonzero(predicted==wanted)[:2]:
            row=values[pos]; order=np.argsort(row); neg=order[:10]; posi=order[-10:][::-1]
            examples.append({'sample_row':int(indices[pos]),'actual_label':int(meta.iloc[indices[pos]]['label']),'predicted_class':int(wanted),'probability_bot':float(probs[pos]),'threshold':threshold,'top_toward_bot':json.dumps([{'feature':names[i],'shap':float(row[i])} for i in posi]),'top_toward_genuine':json.dumps([{'feature':names[i],'shap':float(row[i])} for i in neg])})
            exp=shap.Explanation(values=row,base_values=base,data=dense[pos],feature_names=names); shap.plots.waterfall(exp,max_display=15,show=False); plt.tight_layout(); plt.savefig(water/f'sample_{int(indices[pos])}.png',dpi=160); plt.close()
    pd.DataFrame(examples).to_csv(out/'individual_explanations.csv',index=False)
    # Locked-test descriptive error analysis; it does not alter selection or threshold.
    test=np.flatnonzero(meta['split'].eq('test')); yt=meta.loc[test,'label'].astype(int).to_numpy(); pt=model.predict_proba(X[test])[:,1]; pred=(pt>=threshold).astype(int)
    errors=[]; error_counts={}; error_shap_shapes=[]; error_shap_max_residual=0.0
    for kind,mask in [('false_positive',(yt==0)&(pred==1)),('false_negative',(yt==1)&(pred==0))]:
        error_counts[kind]=int(mask.sum())
        error_positions=np.flatnonzero(mask)
        if len(error_positions)>30:
            rng=np.random.default_rng(SEED+(0 if kind=='false_positive' else 1))
            take=np.sort(rng.choice(error_positions,size=30,replace=False))
        else:
            take=error_positions
        if len(take):
            error_matrix=X[test[take]]
            ev=np.asarray(explainer.shap_values(error_matrix),dtype=float)
            if ev.shape!=(len(take),len(names)) or not np.isfinite(ev).all(): raise RuntimeError(f'Invalid error-analysis SHAP output for {kind}')
            error_expected=np.asarray(model.get_booster().predict(__import__('xgboost').DMatrix(error_matrix),output_margin=True))
            error_residual=np.abs(ev.sum(axis=1)+base-error_expected)
            if not np.isfinite(error_residual).all() or error_residual.max()>1e-4: raise RuntimeError(f'Error-analysis SHAP additivity failed: {error_residual.max()}')
            error_shap_shapes.append(list(ev.shape)); error_shap_max_residual=max(error_shap_max_residual,float(error_residual.max()))
            mean=ev.mean(axis=0); ids=np.argsort(np.abs(mean))[-20:][::-1]
            for i in ids: errors.append({'error_type':kind,'error_count':int(mask.sum()),'analysis_sample_count':int(len(take)),'feature':names[i],'mean_shap':float(mean[i]),'mean_abs_shap':float(abs(mean[i]))})
    pd.DataFrame(errors).to_csv(out/'error_analysis.csv',index=False)
    after=fingerprints(); metrics_after=after['official_test_metrics']
    if before!=after: raise RuntimeError('Input artifact fingerprint changed during Phase 5')
    if metrics_before!=metrics_after: raise RuntimeError('Phase 4 test metrics changed during Phase 5')
    phase4_metrics=pd.read_csv(PHASE4/'official_test_metrics.csv')
    report_errors=pd.DataFrame(errors)
    global_top=imp.head(20).copy(); global_top['mean_abs_shap']=global_top['mean_abs_shap'].map(lambda x:f'{x:.6f}')
    group_report=group_df.copy(); group_report['mean_abs_shap']=group_report['mean_abs_shap'].map(lambda x:f'{x:.6f}')
    test_metric_columns=['accuracy','precision_bot','recall_bot','f1_bot','roc_auc','pr_auc','tn','fp','fn','tp']
    test_report=phase4_metrics[test_metric_columns].copy()
    for column in test_metric_columns[:6]: test_report[column]=test_report[column].map(lambda x:f'{x:.6f}')
    error_findings=[]
    for kind in ('false_positive','false_negative'):
        selected_errors=report_errors[report_errors.error_type.eq(kind)].head(5)
        if len(selected_errors):
            error_findings.append(f"### {kind.replace('_',' ').title()} ({error_counts[kind]})\n\n"
                f"Mean signed SHAP below is computed over {int(selected_errors.analysis_sample_count.iloc[0])} reproducibly sampled errors; positive values move toward the bot class.\n\n"
                f"{markdown_table(selected_errors[['feature','mean_shap','mean_abs_shap']].assign(mean_shap=lambda d:d.mean_shap.map(lambda x:f'{x:.6f}'),mean_abs_shap=lambda d:d.mean_abs_shap.map(lambda x:f'{x:.6f}')))}")
        else:
            error_findings.append(f"### {kind.replace('_',' ').title()} (0)\n\nNo such errors occurred at the locked threshold.")
    validation={'model_type':str(type(model)),'model_inputs':int(model.n_features_in_),'feature_names':len(names),'sample_rows':len(indices),'shap_shape':list(values.shape),'shap_all_finite':bool(np.isfinite(values).all()),'error_shap_shapes':error_shap_shapes,'error_shap_all_finite':True,'shap_output':'raw XGBoost margin','max_additivity_residual':float(max(residual.max(),error_shap_max_residual)),'error_analysis_max_additivity_residual':error_shap_max_residual,'input_fingerprints_before':before,'input_fingerprints_after':after,'phase4_fingerprint_match':True,'selection_lock_sha256_before':before['selection_lock'],'selection_lock_sha256_after':after['selection_lock'],'phase4_test_metrics_sha256_before':metrics_before,'phase4_test_metrics_sha256_after':metrics_after,'locked_threshold':threshold,'test_error_counts':error_counts}
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    report=f'''# Phase 5 — SHAP Explainability Report\n\n## Provenance\nExplained the locked Phase 4 candidate `{MODEL.name}` (SHA-256 `{before['model']}`). The model and reconstructed CSR matrix each have {len(names)} ordered feature inputs. All five Phase 3 files matched Phase 4 run03 provenance; their hashes, the selected model, selection lock, and official test-metrics file were unchanged across this run.\n\n## Method\nTreeExplainer used raw XGBoost margins. A reproducible stratified development sample of {len(indices)} accounts (seed {SEED}) was explained. Full data remained CSR; only the explanation sample was densified for plots. SHAP output shape: `{list(values.shape)}`. Maximum additivity residual: {residual.max():.3g}. The locked development-selected probability threshold was {threshold:.6f}.\n\n## Official Phase 4 test results\nThese are the existing locked-model metrics, not values used to tune this explanation run.\n\n{markdown_table(test_report)}\n\n## Top 20 global features\nRanked by mean absolute SHAP over the development explanation sample. SHAP indicates model contribution, not causality.\n\n{markdown_table(global_top[['feature','mean_abs_shap','feature_group']])}\n\n## Aggregate feature-group importance\nSum of per-feature mean absolute SHAP values over the same sample.\n\n{markdown_table(group_report)}\n\n## False-positive and false-negative findings\nDescriptive analysis of the already locked official test predictions; it did not change selection or threshold. Error totals cover the complete test split. Attribution summaries use a reproducible sample of up to 30 errors per type. Positive signed mean SHAP values move toward the bot class.\n\n{chr(10).join(error_findings)}\n\n## Other outputs\nIndividual explanations omit account IDs. See `individual_explanations.csv`, `global_shap_importance.csv`, and `plots/` for per-row details and visualizations.\n\n## Limitations\nText overlap can inflate benchmark performance for exposed cohorts; it does not by itself prove leakage or memorization. Sparse lexical features are high-dimensional, correlated, and may give unstable individual token attributions; mean absolute SHAP is specific to the sampled development cohort. Network features are transductive topology-only signals and are not causal evidence. Error-attribution averages use at most 30 randomly sampled errors per class and should be treated as descriptive, not a population explanation.\n\n## Validation\nFeature order and model input dimensions matched; matrices remained CSR; SHAP values were finite; raw-margin additivity met the 1e-4 tolerance; feature/global/group outputs were generated; and Phase 3, model, selection-lock, and test-metric hashes remained unchanged.\n'''
    (out/'phase5_explainability_report.md').write_text(report,encoding='utf-8')
if __name__=='__main__': main()
