"""Single-clip extractors matching the existing audio branch's feature definitions."""
import numpy as np

def extract(path, kind):
    if kind == 'opensmile':
        import opensmile
        smile=opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.Functionals)
        return smile.process_file(str(path)).iloc[0].to_dict()
    import librosa
    y,sr=librosa.load(path,sr=16000)
    if not len(y): raise ValueError(f'Empty recording: {path}')
    duration=len(y)/sr
    rms=librosa.feature.rms(y=y)[0]
    zcr=librosa.feature.zero_crossing_rate(y)[0]
    centroid=librosa.feature.spectral_centroid(y=y,sr=sr)[0]
    pitch=librosa.yin(y,fmin=65,fmax=400,sr=sr)
    voiced=pitch[(pitch>=65)&(pitch<=400)]
    intervals=librosa.effects.split(y,top_db=25)
    speech=sum(end-start for start,end in intervals)/sr
    return dict(duration=duration,pitch_mean=np.mean(voiced) if len(voiced) else 0,pitch_std=np.std(voiced) if len(voiced) else 0,energy_mean=np.mean(rms),energy_std=np.std(rms),zcr_mean=np.mean(zcr),spectral_centroid_mean=np.mean(centroid),speech_duration=speech,pause_duration=max(0,duration-speech),speech_ratio=speech/duration,pause_count=max(0,len(intervals)-1))
