import librosa
import numpy as np

def getmetadata(filename):
    # Load the audio file
    y, sr = librosa.load(filename)

    # Calculate onset strength
    onset_env = librosa.onset.onset_strength(y=y)

    # Calculate tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Calculate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Calculate chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # Calculate RMSE
    rmse = librosa.feature.rms(y=y)

    # Calculate spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Calculate spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Calculate spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Calculate zero crossing rate
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    # Calculate MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Metadata dictionary
    metadata_dict = {
        'tempo': tempo,
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spec_centroid),
        'spectral_bandwidth': np.mean(spec_bw),
        'rolloff': np.mean(spec_rolloff),
        'zero_crossing_rates': np.mean(zero_crossing)
    }

    # Add MFCCs to the dictionary
    for i in range(1, 21):
        metadata_dict[f'mfcc{i}'] = np.mean(mfcc[i - 1])

    return list(metadata_dict.values())
