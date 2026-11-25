"""Nested speaker-independent fusion; no test labels used for model selection."""
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score

class OptimizedFusion:
    def __init__(self, seed=42): self.seed=seed
    def fit(self,t,a,y,groups):
        t,a,y,groups=map(np.asarray,(t,a,y,groups))
        for n in range(min(3,len(np.unique(groups))),1,-1):
            splits=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=self.seed).split(t,y,groups))
            if all(len(np.unique(y[tr]))==2 for tr,va in splits): break
        else: raise ValueError('Not enough speaker groups for nested validation')
        cap=min(len(tr) for tr,va in splits)-1
        dims=sorted(set(min(k,cap,t.shape[1]) for k in (32,64,128)))
        lr=lambda c: LogisticRegression(C=c,max_iter=2000,random_state=self.seed)
        text=[make_pipeline(StandardScaler(),lr(c)) for c in (.01,.1,1)]
        text += [make_pipeline(StandardScaler(),PCA(n_components=k,svd_solver='randomized',random_state=self.seed),lr(c)) for k in dims for c in (.1,1)]
        audio=[make_pipeline(StandardScaler(),SVC(C=c,gamma=g)) for c in (.1,1,10) for g in ('scale',.01,.001)]
        audio += [make_pipeline(StandardScaler(),lr(c)) for c in (.01,.1,1)]
        def select(candidates,x):
            best=None
            for est in candidates:
                scores=np.empty(len(y))
                for tr,va in splits:
                    scores[va]=clone(est).fit(x[tr],y[tr]).decision_function(x[va])
                # Choose by average validation-fold accuracy, matching outer metric.
                acc=np.mean([accuracy_score(y[va],scores[va]>=0) for tr,va in splits])
                if best is None or acc>best[0]: best=(acc,est,scores.copy())
            _,est,scores=best
            # Calibration uses training-only out-of-fold scores, not SVC internal random CV.
            cal=LogisticRegression().fit(scores.reshape(-1,1),y)
            return clone(est).fit(x,y),cal,scores
        self.text,self.tc,ts=select(text,t)
        self.audio,self.ac,az=select(audio,a)
        # Approximate inner cross-fit calibration; candidate selection shares
        # inner folds. The independent outer test fold evaluates the whole procedure.
        def calibrated(s):
            p=np.empty(len(y))
            for tr,va in splits:
                c=LogisticRegression().fit(s[tr,None],y[tr]); p[va]=c.predict_proba(s[va,None])[:,1]
            return p
        tp,ap=calibrated(ts),calibrated(az)
        best=(-1,0,0.5)
        for w in np.linspace(0,1,11):
            p=w*tp+(1-w)*ap
            for th in (.4,.45,.5,.55,.6):
                acc=np.mean([accuracy_score(y[va],p[va]>=th) for tr,va in splits])
                if acc>best[0]: best=(acc,float(w),th)
        self.validation_accuracy,self.text_weight,self.threshold=best
        return self
    def probabilities(self,t,a):
        tp=self.tc.predict_proba(self.text.decision_function(t).reshape(-1,1))[:,1]
        ap=self.ac.predict_proba(self.audio.decision_function(a).reshape(-1,1))[:,1]
        return tp,ap,self.text_weight*tp+(1-self.text_weight)*ap
