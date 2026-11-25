"""Fixed early and late fusion baselines."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class MultimodalFusionModel:
    def __init__(self, mode='early', seed=42):
        self.mode = mode
        self.text = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
        self.audio = make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed))
        self.early = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
    def fit(self, text, audio, y):
        if self.mode == 'early': self.early.fit(np.column_stack([text,audio]), y)
        else:
            if self.mode in ('text','late'): self.text.fit(text,y)
            if self.mode in ('audio','late'): self.audio.fit(audio,y)
        return self
    def predict_proba(self,text,audio):
        if self.mode == 'early': return self.early.predict_proba(np.column_stack([text,audio]))
        if self.mode == 'text': return self.text.predict_proba(text)
        if self.mode == 'audio': return self.audio.predict_proba(audio)
        return (self.text.predict_proba(text)+self.audio.predict_proba(audio))/2
