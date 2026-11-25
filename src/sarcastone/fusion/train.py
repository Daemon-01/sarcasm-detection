"""Strict ID alignment and evaluation using original frozen folds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, RocCurveDisplay, ConfusionMatrixDisplay
from sarcastone.eval.metrics import compute_classification_metrics, aggregate_fold_metrics
from sarcastone.data.folds import load_frozen_folds
from .models import MultimodalFusionModel

FILES={'prosody':'features/prosody/prosody_features.csv','opensmile':'features/emotion/opensmile_emotion_features.csv'}
SEEDS=[42,123,2026]

def load_inputs(manifest, kinds):
    df=pd.read_csv(manifest,dtype={'utterance_id':str,'target_key':str})
    if df.utterance_id.duplicated().any() or df.target_key.duplicated().any(): raise ValueError('Duplicate manifest IDs')
    with np.load('features/text/roberta_embeddings_utterance_only.npz',allow_pickle=False) as z:
        text=np.stack([z[k] for k in df.utterance_id])
    if text.shape != (len(df),768): raise ValueError('Unexpected text feature shape')
    blocks=[]; columns={}
    for kind in kinds:
        table=pd.read_csv(FILES[kind],dtype={'KEY':str}).set_index('KEY',verify_integrity=True)
        missing=set(df.target_key)-set(table.index)
        if missing: raise ValueError(f'{kind}: missing {len(missing)} target clips; examples {sorted(missing)[:5]}')
        columns[kind]=list(table.columns)
        blocks.append(table.loc[df.target_key].to_numpy(dtype=float))
    audio=np.column_stack(blocks)
    if not np.isfinite(text).all() or not np.isfinite(audio).all(): raise ValueError('Nonfinite features; inspect source recordings/features')
    return df,text,audio,columns

def train_fusion_model(manifest='data/manifest.csv', folds='data/folds.json', kinds=('prosody','opensmile'), output='results/multimodal'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    df,t,a,columns=load_inputs(manifest,kinds)
    frozen=load_frozen_folds(folds,manifest)
    if len(frozen['folds'])!=5: raise ValueError('Expected five frozen folds')
    y=df.sarcasm.to_numpy(dtype=int); ids=df.utterance_id
    if set(y)!={0,1}: raise ValueError('Expected binary labels')
    splits=[]
    for f in frozen['folds']:
        tr=np.flatnonzero(ids.isin(f['train_utterance_ids'])); te=np.flatnonzero(ids.isin(f['test_utterance_ids']))
        if len(tr)+len(te)!=len(df) or set(y[tr])!={0,1}: raise ValueError('Invalid fold partition')
        splits.append((f['fold'],tr,te))
    summary=[]
    for mode in ('text','audio','early','late'):
        rows=[]; predictions=[]; metrics=[]
        for seed in SEEDS:
            for fold,tr,te in splits:
                model=MultimodalFusionModel(mode,seed).fit(t[tr],a[tr],y[tr])
                p=model.predict_proba(t[te],a[te])[:,1]; pred=(p>=.5).astype(int)
                m=compute_classification_metrics(y[te],pred,p); metrics.append(m)
                rows.append(dict(seed=seed,fold=fold,**{k:v for k,v in m.items() if k!='confusion_matrix'},**m['confusion_matrix']))
                predictions.extend(dict(seed=seed,fold=fold,utterance_id=ids.iloc[i],y_true=int(y[i]),y_pred=int(q),y_prob=float(r)) for i,q,r in zip(te,pred,p))
        pd.DataFrame(rows).to_csv(out/f'{mode}_folds.csv',index=False)
        pr=pd.DataFrame(predictions); pr.to_csv(out/f'{mode}_predictions.csv',index=False)
        agg=aggregate_fold_metrics(metrics); summary.append(dict(model=mode,**agg))
        first=pr[pr.seed==SEEDS[0]]
        cm=confusion_matrix(first.y_true,first.y_pred,labels=[0,1]); np.savetxt(out/f'{mode}_confusion.csv',cm,fmt='%d',delimiter=',')
        ConfusionMatrixDisplay(cm,display_labels=['Non-sarcastic','Sarcastic']).plot(); plt.tight_layout(); plt.savefig(out/f'{mode}_confusion.png'); plt.close()
        RocCurveDisplay.from_predictions(first.y_true,first.y_prob); plt.tight_layout(); plt.savefig(out/f'{mode}_roc.png'); plt.close()
        (out/f'{mode}_report.json').write_text(json.dumps(classification_report(first.y_true,first.y_pred,labels=[0,1],output_dict=True,zero_division=0),indent=2))
        if mode in ('early','late'):
            joblib.dump({'model':MultimodalFusionModel(mode).fit(t,a,y),'columns':columns,'kinds':list(kinds)},out/f'{mode}.joblib')
    result=pd.DataFrame(summary); result.to_csv(out/'comparison.csv',index=False); print(result.to_string(index=False)); return result
