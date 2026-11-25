"""Run optimized fusion without overwriting the original experiments."""
import argparse,sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); os.chdir(ROOT)
import numpy as np
import pandas as pd
import joblib
from sarcastone.fusion.train import load_inputs, SEEDS
from sarcastone.fusion.optimized import OptimizedFusion
from sarcastone.data.folds import load_frozen_folds
from sarcastone.eval.metrics import compute_classification_metrics,aggregate_fold_metrics
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,classification_report

def train(kinds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path('results/multimodal_improved'); out.mkdir(parents=True,exist_ok=True)
    df,t,a,columns=load_inputs('data/manifest.csv',kinds)
    y=df.sarcasm.to_numpy(dtype=int); groups=df.target_speaker.astype(str).to_numpy()
    frozen=load_frozen_folds('data/folds.json','data/manifest.csv')
    modes=('tuned_text','tuned_audio','weighted_fusion'); rows={m:[] for m in modes}; preds={m:[] for m in modes}; metrics={m:[] for m in modes}
    for seed in SEEDS:
        for f in frozen['folds']:
            tr=np.flatnonzero(df.utterance_id.isin(f['train_utterance_ids'])); te=np.flatnonzero(df.utterance_id.isin(f['test_utterance_ids']))
            if len(tr)+len(te)!=len(df) or set(groups[tr]) & set(groups[te]): raise ValueError('Invalid outer fold')
            print(f"Seed {seed}, fold {f['fold']}: tuning...",flush=True)
            model=OptimizedFusion(seed).fit(t[tr],a[tr],y[tr],groups[tr])
            for mode,p,th in zip(modes,model.probabilities(t[te],a[te]),(.5,.5,model.threshold)):
                q=(p>=th).astype(int); m=compute_classification_metrics(y[te],q,p); metrics[mode].append(m)
                rows[mode].append(dict(seed=seed,fold=f['fold'],text_weight=model.text_weight,threshold=th,**{k:v for k,v in m.items() if k!='confusion_matrix'},**m['confusion_matrix']))
                preds[mode].extend(dict(seed=seed,fold=f['fold'],utterance_id=df.utterance_id.iloc[i],y_true=int(y[i]),y_pred=int(j),y_prob=float(k)) for i,j,k in zip(te,q,p))
            print(f'  Text weight={model.text_weight:.2f}; threshold={model.threshold:.2f}; fusion accuracy={metrics[modes[-1]][-1]["accuracy"]:.4f}',flush=True)
    summary=[]
    for mode in modes:
        pd.DataFrame(rows[mode]).to_csv(out/f'{mode}_folds.csv',index=False)
        pr=pd.DataFrame(preds[mode]); pr.to_csv(out/f'{mode}_predictions.csv',index=False); pr=pr[pr.seed==SEEDS[0]]
        summary.append(dict(model=mode,**aggregate_fold_metrics(metrics[mode])))
        ConfusionMatrixDisplay.from_predictions(pr.y_true,pr.y_pred,labels=[0,1]); plt.tight_layout(); plt.savefig(out/f'{mode}_confusion.png'); plt.close()
        RocCurveDisplay.from_predictions(pr.y_true,pr.y_prob); plt.tight_layout(); plt.savefig(out/f'{mode}_roc.png'); plt.close()
        (out/f'{mode}_report.json').write_text(json.dumps(classification_report(pr.y_true,pr.y_pred,output_dict=True,zero_division=0),indent=2))
    result=pd.DataFrame(summary); result.to_csv(out/'comparison.csv',index=False)
    print(result[['model','accuracy_mean','accuracy_std','f1_macro_mean','roc_auc_mean']].to_string(index=False))
    print('Fitting deployment model on all samples...',flush=True)
    joblib.dump(dict(model=OptimizedFusion().fit(t,a,y,groups),kinds=kinds,columns=columns),out/'optimized.joblib')

def predict(text,audio):
    from sarcastone.text.embed import RobertaFeatureExtractor
    from sarcastone.text.preprocess import prepare_target_text
    from sarcastone.fusion.audio_features import extract
    b=joblib.load('results/multimodal_improved/optimized.joblib'); m=b['model']
    t=RobertaFeatureExtractor().extract_embeddings([prepare_target_text(text)])
    a=[]
    for kind in b['kinds']:
        vals=extract(audio,kind); a.extend(vals[c] for c in b['columns'][kind])
    tp,ap,p=m.probabilities(t,np.asarray(a).reshape(1,-1))
    print('Prediction:', 'Sarcastic' if p[0]>=m.threshold else 'Non-sarcastic')
    print(f'Text={tp[0]:.4f}, Audio={ap[0]:.4f}, Combined={p[0]:.4f}, Threshold={m.threshold:.2f}')

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--audio',choices=['both','prosody','opensmile'],default='both')
    pr=sub.add_parser('predict'); pr.add_argument('--text',required=True); pr.add_argument('--audio',required=True)
    args=p.parse_args()
    if args.cmd=='train': train(('prosody','opensmile') if args.audio=='both' else (args.audio,))
    else: predict(args.text,args.audio)
