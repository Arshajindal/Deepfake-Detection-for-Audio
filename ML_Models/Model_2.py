import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time
from typing import Dict, List, Any, Tuple
import os

# -------------------- AIVoiceDetector --------------------

class AIVoiceDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models (currently placeholders)."""
        model_dir = "C:/Users/GOWTHAM/OneDrive/Desktop/Hackathon/Deepfake-Detection-for-Audio/ML_Models"
        
        # Placeholder RandomForest models
        self.models['spectral'] = self._create_spectral_classifier()
        self.models['prosodic'] = self._create_prosodic_classifier()
        self.models['neural'] = self._create_neural_classifier()
        
        # Ensemble (not trained yet, just placeholder)
        self.ensemble = VotingClassifier([
            ('spectral', self.models['spectral']),
            ('prosodic', self.models['prosodic']),
            ('neural', self.models['neural'])
        ], voting='soft')
        
    def _create_spectral_classifier(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _create_prosodic_classifier(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _create_neural_classifier(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def detect(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Main detection function"""
        start_time = time.time()
        
        # Extract condensed feature sets
        spectral_features = self._analyze_spectral_features(features)
        prosodic_features = self._analyze_prosodic_features(features)
        neural_features = self._analyze_neural_artifacts(features)
        
        # Compute heuristic scores
        spectral_score = self._get_spectral_score(spectral_features)
        prosodic_score = self._get_prosodic_score(prosodic_features)
        neural_score = self._get_neural_score(neural_features)
        
        # Final average
        final_score = (spectral_score + prosodic_score + neural_score) / 3
        is_ai_generated = final_score > 0.3
        
        processing_time = time.time() - start_time
        
        return {
            "is_ai_generated": is_ai_generated,
            "confidence_score": float(final_score),
            "detailed_analysis": {
                "spectral_analysis": {
                    "score": float(spectral_score),
                    "indicators": self._get_spectral_indicators(spectral_features)
                },
                "prosodic_analysis": {
                    "score": float(prosodic_score), 
                    "indicators": self._get_prosodic_indicators(prosodic_features)
                },
                "neural_artifacts": {
                    "score": float(neural_score),
                    "indicators": self._get_neural_indicators(neural_features)
                }
            },
            "processing_time": processing_time
        }
    
    # ---------------- Spectral analysis ----------------
    def _analyze_spectral_features(self, features: Dict) -> Dict:
        mfccs = features['mfccs']
        spectral_centroid = features['spectral_centroid']
        spectral_rolloff = features['spectral_rolloff']
        
        mfcc_variance = np.var(mfccs, axis=1)
        spectral_flatness = self._calculate_spectral_flatness(features['stft'])
        
        return {
            'mfcc_variance': mfcc_variance,
            'spectral_flatness': spectral_flatness,
            'centroid_std': np.std(spectral_centroid),
            'rolloff_std': np.std(spectral_rolloff)
        }
    
    # ---------------- Prosodic analysis ----------------
    def _analyze_prosodic_features(self, features: Dict) -> Dict:
        f0 = features.get('f0', np.array([]))
        
        if len(f0) > 0:
            f0_clean = f0[f0 > 0]  # remove unvoiced
            
            return {
                'f0_mean': np.mean(f0_clean) if len(f0_clean) > 0 else 0,
                'f0_std': np.std(f0_clean) if len(f0_clean) > 0 else 0,
                'f0_range': np.ptp(f0_clean) if len(f0_clean) > 0 else 0,
                'voiced_ratio': len(f0_clean) / len(f0) if len(f0) > 0 else 0,
                'f0_contour_complexity': self._calculate_f0_complexity(f0_clean) if len(f0_clean) > 0 else 0
            }
        
        return {'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 'voiced_ratio': 0, 'f0_contour_complexity': 0}
    
    # ---------------- Neural artifacts analysis ----------------
    def _analyze_neural_artifacts(self, features: Dict) -> Dict:
        mfccs = features['mfccs']
        stft = features['stft']
        
        high_freq_energy = np.mean(np.abs(stft[stft.shape[0]//2:, :]))
        low_freq_energy = np.mean(np.abs(stft[:stft.shape[0]//2, :]))
        
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        autocorr = np.correlate(mfccs[0], mfccs[0], mode='full')
        periodicity = np.max(autocorr[len(autocorr)//2+1:]) / (np.max(autocorr) + 1e-8)
        
        return {
            'freq_energy_ratio': freq_ratio,
            'periodicity_score': periodicity,
            'mfcc_smoothness': self._calculate_smoothness(mfccs),
            'spectral_peaks': self._count_spectral_peaks(stft)
        }
    
    # ---------------- Rule-based scorers ----------------
    def _get_spectral_score(self, features: Dict) -> float:
        score = 0.0
        if np.mean(features['mfcc_variance']) < 0.5:
            score += 0.4
        if features['spectral_flatness'] > 0.8 or features['spectral_flatness'] < 0.2:
            score += 0.2
        if features['centroid_std'] < 200:
            score += 0.25
        if features['rolloff_std'] < 300:
            score += 0.25
        return min(score, 1.0)
    
    def _get_prosodic_score(self, features: Dict) -> float:
        score = 0.0
        if features['f0_std'] < 10:
            score += 0.3
        if features['f0_range'] < 50:
            score += 0.2
        if features['voiced_ratio'] > 0.9 or features['voiced_ratio'] < 0.3:
            score += 0.3
        if features['f0_contour_complexity'] < 0.1:
            score += 0.2
        return min(score, 1.0)
    
    def _get_neural_score(self, features: Dict) -> float:
        score = 0.0
        if features['freq_energy_ratio'] < 0.1 or features['freq_energy_ratio'] > 2.0:
            score += 0.3
        if features['periodicity_score'] > 0.8:
            score += 0.25
        if features['mfcc_smoothness'] > 0.9:
            score += 0.25
        if features['spectral_peaks'] < 5 or features['spectral_peaks'] > 50:
            score += 0.2
        return min(score, 1.0)
    
    def _get_zcr_score(self, features):
        score, indicators = 0.0, []
        if features['zcr'].mean() < 0.01:
            score += 0.25
            indicators.append("Unnaturally low zero-crossing rate (AI trait)")
        elif features['zcr'].mean() > 0.2:
            score += 0.15
            indicators.append("Noisy/high ZCR (possible AI artifact)")
        else:
            indicators.append("Normal zero-crossing rate")
        return score, indicators

    def _get_tempo_score(self, features):
        score, indicators = 0.0, []
        if features['tempo'] > 250 or features['tempo'] < 50:
            score += 0.25
            indicators.append("Unnatural speaking tempo (AI trait)")
        else:
            indicators.append("Normal tempo characteristics")
        return score, indicators
    
    # ---------------- Helper calculations ----------------
    def _calculate_spectral_flatness(self, stft: np.ndarray) -> float:
        magnitude = np.abs(stft)
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
        arithmetic_mean = np.mean(magnitude, axis=0)
        return float(np.mean(geometric_mean / (arithmetic_mean + 1e-10)))
    
    def _calculate_f0_complexity(self, f0: np.ndarray) -> float:
        if len(f0) < 2:
            return 0.0
        diff2 = np.diff(np.diff(f0))
        return float(np.std(diff2) / (np.mean(f0) + 1e-8))
    
    def _calculate_smoothness(self, mfccs: np.ndarray) -> float:
        scores = []
        for coeff in mfccs:
            diff = np.diff(coeff)
            scores.append(1.0 / (1.0 + np.std(diff)))
        return float(np.mean(scores))
    
    def _count_spectral_peaks(self, stft: np.ndarray) -> int:
        magnitude = np.abs(stft)
        avg_spectrum = np.mean(magnitude, axis=1)
        peaks = 0
        for i in range(1, len(avg_spectrum)-1):
            if (avg_spectrum[i] > avg_spectrum[i-1] and 
                avg_spectrum[i] > avg_spectrum[i+1] and
                avg_spectrum[i] > np.mean(avg_spectrum)):
                peaks += 1
        return peaks
    
    # ---------------- Human-readable indicators ----------------
    def _get_spectral_indicators(self, features: Dict) -> List[str]:
        indicators = []
        if np.mean(features['mfcc_variance']) < 0.5:
            indicators.append("Low spectral variance detected")
        if features['spectral_flatness'] > 0.8:
            indicators.append("High spectral flatness (noise-like)")
        elif features['spectral_flatness'] < 0.2:
            indicators.append("Low spectral flatness (tonal)")
        if features['centroid_std'] < 200:
            indicators.append("Consistent spectral centroid")
        if features['rolloff_std'] < 300:
            indicators.append("Consistent spectral rolloff")
        return indicators or ["Normal spectral characteristics"]
    
    def _get_prosodic_indicators(self, features: Dict) -> List[str]:
        indicators = []
        if features['f0_std'] < 10:
            indicators.append("Monotone speech pattern")
        if features['f0_range'] < 50:
            indicators.append("Narrow pitch range")
        if features['voiced_ratio'] > 0.9:
            indicators.append("Unusually high voiced speech ratio")
        elif features['voiced_ratio'] < 0.3:
            indicators.append("Unusually low voiced speech ratio")
        if features['f0_contour_complexity'] < 0.1:
            indicators.append("Simple pitch contour")
        return indicators or ["Normal prosodic characteristics"]
    
    def _get_neural_indicators(self, features: Dict) -> List[str]:
        indicators = []
        if features['freq_energy_ratio'] < 0.1:
            indicators.append("Low high-frequency content")
        elif features['freq_energy_ratio'] > 2.0:
            indicators.append("High high-frequency content")
        if features['periodicity_score'] > 0.8:
            indicators.append("Highly periodic signal")
        if features['mfcc_smoothness'] > 0.9:
            indicators.append("Very smooth MFCC transitions")
        if features['spectral_peaks'] < 5:
            indicators.append("Few spectral peaks")
        elif features['spectral_peaks'] > 50:
            indicators.append("Many spectral peaks")
        return indicators or ["Normal signal characteristics"]


import matplotlib.pyplot as plt
import librosa.display
import base64   
from io import BytesIO

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        stft = librosa.stft(audio_data)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=self.n_mfcc)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        f0 = librosa.yin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        return {
            'stft': stft,
            'mfccs': mfccs,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zcr': zcr,
            'f0': f0,
            'chroma': chroma,
            'tempo': tempo,
            'beats': beats
        }

# -------------------- Dataset Runner --------------------

from pathlib import Path
import pandas as pd

def process_folder(folder_path: Path, label: str, audio_processor, detector):
    results = []
    for file in folder_path.glob("*"):
        if file.suffix.lower() == ".wav":
            print(f"Processing {file} ...")
            try:    
                # Load audio
                audio_data, sr = librosa.load(file, sr=audio_processor.sample_rate)
                # Extract features
                features = audio_processor.extract_features(audio_data, sr)
                # Detect
                result = detector.detect(features)
                # Store
                results.append({
                    "file": str(file),
                    "label": label,
                    "prediction": "fake" if result["is_ai_generated"] else "real",
                    "confidence": result["confidence_score"],
                    "details": result["detailed_analysis"]
                })
            except Exception as e:
                print(f"Error processing {file}: {e}")
    return results

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    detector = AIVoiceDetector()

    real_path = Path("C:\\Users\\GOWTHAM\\OneDrive\\Desktop\\Hackathon\\Deepfake-Detection-for-Audio\\data\\Dummy Test Dataset\\testing\\real")
    fake_path = Path("C:\\Users\\GOWTHAM\\OneDrive\\Desktop\\Hackathon\\Deepfake-Detection-for-Audio\\data\\Dummy Test Dataset\\testing\\fake")

    real_results = process_folder(real_path, "real", audio_processor, detector)
    fake_results = process_folder(fake_path, "fake", audio_processor, detector)
    
    all_results = real_results + fake_results
    
    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("detection_results.csv", index=False)
        print("✅ Saved results to detection_results.csv")
    else:
        print("⚠️ No audio files processed. Check your folder paths and file extensions.")

    
