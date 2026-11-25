import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import os
os.chdir(ROOT)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--csv',default='data/mustard++_text.csv'); prep.add_argument('--audio-dir',default='data/processed/audio')
    ex=sub.add_parser('extract'); ex.add_argument('--audio-dir',default='data/processed/audio')
    tr=sub.add_parser('train'); tr.add_argument('--audio',choices=['prosody','opensmile','both'],default='both')
    pred=sub.add_parser('predict'); pred.add_argument('--text',required=True); pred.add_argument('--audio',required=True); pred.add_argument('--mode',choices=['early','late'],default='early')
    args=p.parse_args()
    if args.cmd=='prepare':
        from sarcastone.data.manifest import build_and_save_manifest
        build_and_save_manifest(args.csv,audio_dir=args.audio_dir)
    elif args.cmd=='extract':
        from sarcastone.text.embed import export_roberta_embeddings
        from sarcastone.fusion.audio_features import extract
        from sarcastone.fusion.train import FILES
        df=pd.read_csv('data/manifest.csv',dtype={'target_key':str})
        export_roberta_embeddings()
        for kind,path in FILES.items():
            rows=[]
            for i,key in enumerate(df.target_key):
                rows.append(dict(KEY=key,**extract(Path(args.audio_dir)/f'{key}.wav',kind)))
                if (i+1)%100==0: print(kind,i+1)
            Path(path).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(path,index=False)
    elif args.cmd=='train':
        from sarcastone.fusion.train import train_fusion_model
        train_fusion_model(kinds=('prosody','opensmile') if args.audio=='both' else (args.audio,))
    else:
        import joblib
        from sarcastone.text.embed import RobertaFeatureExtractor
        from sarcastone.text.preprocess import prepare_target_text
        from sarcastone.fusion.audio_features import extract
        bundle=joblib.load(f'results/multimodal/{args.mode}.joblib')
        t=RobertaFeatureExtractor().extract_embeddings([prepare_target_text(args.text)])
        a=[]
        for kind in bundle['kinds']:
            values=extract(args.audio,kind); a.extend(values[c] for c in bundle['columns'][kind])
        prob=bundle['model'].predict_proba(t,np.asarray(a).reshape(1,-1))[0]
        print('Prediction:', 'Sarcastic' if prob[1]>=.5 else 'Non-sarcastic'); print('Probabilities:',dict(non_sarcastic=float(prob[0]),sarcastic=float(prob[1])))
if __name__=='__main__': main()
