import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time
from typing import Dict, List, Any, Tuple
import os

class AIVoiceDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models"""
        model_dir = "ml_models/trained_models"
        
        # For MVP, we'll use simple rule-based detection with ML components
        # In production, these would be trained on large datasets
        
        # Spectral analyzer
        self.models['spectral'] = self._create_spectral_classifier()
        self.models['prosodic'] = self._create_prosodic_classifier()
        self.models['neural'] = self._create_neural_classifier()
        
        # Ensemble model
        self.ensemble = VotingClassifier([
            ('spectral', self.models['spectral']),
            ('prosodic', self.models['prosodic']),
            ('neural', self.models['neural'])
        ], voting='soft')
        
    def _create_spectral_classifier(self):
        """Create spectral analysis classifier"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _create_prosodic_classifier(self):
        """Create prosodic features classifier"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _create_neural_classifier(self):
        """Create neural artifacts classifier"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def detect(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Main detection function
        """
        start_time = time.time()
        
        # Extract different types of features
        spectral_features = self._analyze_spectral_features(features)
        prosodic_features = self._analyze_prosodic_features(features)
        neural_features = self._analyze_neural_artifacts(features)
        
        # Individual model predictions
        spectral_score = self._get_spectral_score(spectral_features)
        prosodic_score = self._get_prosodic_score(prosodic_features)
        neural_score = self._get_neural_score(neural_features)
        
        # Ensemble prediction
        final_score = (spectral_score + prosodic_score + neural_score) / 3
        is_ai_generated = final_score > 0.5
        
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
    
    def _analyze_spectral_features(self, features: Dict) -> Dict:
        """Analyze spectral characteristics"""
        mfccs = features['mfccs']
        spectral_centroid = features['spectral_centroid']
        spectral_rolloff = features['spectral_rolloff']
        
        # Calculate spectral irregularities
        mfcc_variance = np.var(mfccs, axis=1)
        spectral_flatness = self._calculate_spectral_flatness(features['stft'])
        
        return {
            'mfcc_variance': mfcc_variance,
            'spectral_flatness': spectral_flatness,
            'centroid_std': np.std(spectral_centroid),
            'rolloff_std': np.std(spectral_rolloff)
        }
    
    def _analyze_prosodic_features(self, features: Dict) -> Dict:
        """Analyze prosodic characteristics"""
        f0 = features.get('f0', np.array([]))
        
        if len(f0) > 0:
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            
            return {
                'f0_mean': np.mean(f0_clean) if len(f0_clean) > 0 else 0,
                'f0_std': np.std(f0_clean) if len(f0_clean) > 0 else 0,
                'f0_range': np.ptp(f0_clean) if len(f0_clean) > 0 else 0,
                'voiced_ratio': len(f0_clean) / len(f0) if len(f0) > 0 else 0,
                'f0_contour_complexity': self._calculate_f0_complexity(f0_clean) if len(f0_clean) > 0 else 0
            }
        
        return {
            'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 
            'voiced_ratio': 0, 'f0_contour_complexity': 0
        }
    
    def _analyze_neural_artifacts(self, features: Dict) -> Dict:
        """Analyze neural network artifacts"""
        mfccs = features['mfccs']
        stft = features['stft']
        
        # Look for compression artifacts and unnatural patterns
        high_freq_energy = np.mean(np.abs(stft[stft.shape[0]//2:, :]))
        low_freq_energy = np.mean(np.abs(stft[:stft.shape[0]//2, :]))
        
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        # Periodicity analysis
        autocorr = np.correlate(mfccs[0], mfccs[0], mode='full')
        periodicity = np.max(autocorr[len(autocorr)//2+1:]) / np.max(autocorr)
        
        return {
            'freq_energy_ratio': freq_ratio,
            'periodicity_score': periodicity,
            'mfcc_smoothness': self._calculate_smoothness(mfccs),
            'spectral_peaks': self._count_spectral_peaks(stft)
        }
    
    def _get_spectral_score(self, features: Dict) -> float:
        """Calculate spectral-based AI probability"""
        # Simple rule-based scoring for MVP
        score = 0.0
        
        # AI voices often have less spectral variance
        if np.mean(features['mfcc_variance']) < 0.5:
            score += 0.3
            
        # Unnatural spectral flatness
        if features['spectral_flatness'] > 0.8 or features['spectral_flatness'] < 0.2:
            score += 0.2
            
        # Too consistent spectral features
        if features['centroid_std'] < 200:
            score += 0.25
            
        if features['rolloff_std'] < 300:
            score += 0.25
            
        return min(score, 1.0)
    
    def _get_prosodic_score(self, features: Dict) -> float:
        """Calculate prosodic-based AI probability"""
        score = 0.0
        
        # Unnatural F0 patterns
        if features['f0_std'] < 10:  # Too monotone
            score += 0.3
            
        if features['f0_range'] < 50:  # Too narrow range
            score += 0.2
            
        # Unnatural voiced/unvoiced ratio
        if features['voiced_ratio'] > 0.9 or features['voiced_ratio'] < 0.3:
            score += 0.3
            
        # Too simple F0 contour
        if features['f0_contour_complexity'] < 0.1:
            score += 0.2
            
        return min(score, 1.0)
    
    def _get_neural_score(self, features: Dict) -> float:
        """Calculate neural artifacts-based AI probability"""
        score = 0.0
        
        # Unusual frequency distribution
        if features['freq_energy_ratio'] < 0.1 or features['freq_energy_ratio'] > 2.0:
            score += 0.3
            
        # Too periodic
        if features['periodicity_score'] > 0.8:
            score += 0.25
            
        # Too smooth MFCC transitions
        if features['mfcc_smoothness'] > 0.9:
            score += 0.25
            
        # Unusual spectral peak distribution
        if features['spectral_peaks'] < 5 or features['spectral_peaks'] > 50:
            score += 0.2
            
        return min(score, 1.0)
    
    def _calculate_spectral_flatness(self, stft: np.ndarray) -> float:
        """Calculate spectral flatness measure"""
        magnitude = np.abs(stft)
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
        arithmetic_mean = np.mean(magnitude, axis=0)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        return np.mean(flatness)
    
    def _calculate_f0_complexity(self, f0: np.ndarray) -> float:
        """Calculate F0 contour complexity"""
        if len(f0) < 2:
            return 0.0
        
        # Calculate derivative and second derivative
        diff1 = np.diff(f0)
        diff2 = np.diff(diff1)
        
        # Complexity based on variations
        complexity = np.std(diff2) / (np.mean(f0) + 1e-8)
        return complexity
    
    def _calculate_smoothness(self, mfccs: np.ndarray) -> float:
        """Calculate MFCC smoothness"""
        smoothness_scores = []
        for mfcc_coeff in mfccs:
            diff = np.diff(mfcc_coeff)
            smoothness = 1.0 / (1.0 + np.std(diff))
            smoothness_scores.append(smoothness)
        return np.mean(smoothness_scores)
    
    def _count_spectral_peaks(self, stft: np.ndarray) -> int:
        """Count spectral peaks"""
        magnitude = np.abs(stft)
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Simple peak detection
        peaks = 0
        for i in range(1, len(avg_spectrum)-1):
            if (avg_spectrum[i] > avg_spectrum[i-1] and 
                avg_spectrum[i] > avg_spectrum[i+1] and
                avg_spectrum[i] > np.mean(avg_spectrum)):
                peaks += 1
        
        return peaks
    
    def _get_spectral_indicators(self, features: Dict) -> List[str]:
        """Get human-readable spectral indicators"""
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
        """Get human-readable prosodic indicators"""
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
        """Get human-readable neural artifact indicators"""
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

# backend/app/utils/audio_utils.py
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from typing import Dict, Tuple, Any
from fastapi import UploadFile
import tempfile
import os

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        
    async def load_audio(self, file: UploadFile) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data with sample rate"""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load audio using librosa
            audio_data, sr = librosa.load(tmp_path, sr=self.sample_rate)
            return audio_data, sr
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features"""
        
        # STFT
        stft = librosa.stft(audio_data)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=self.n_mfcc)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # Fundamental frequency (F0)
        f0 = librosa.yin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        
        # Tempo and beat
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
    
    def generate_visualizations(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, str]:
        """Generate visualization plots as base64 encoded strings"""
        
        visualizations = {}
        
        # 1. Waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio_data, sr=sample_rate)
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        visualizations['waveform'] = self._plot_to_base64()
        
        # 2. Spectrogram
        plt.figure(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        visualizations['spectrogram'] = self._plot_to_base64()
        
        # 3. MFCCs
        plt.figure(figsize=(12, 6))
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
        plt.colorbar()
        plt.title('MFCC Features')
        plt.ylabel('MFCC Coefficients')
        visualizations['mfcc'] = self._plot_to_base64()
        
        # 4. Spectral features
        plt.figure(figsize=(12, 8))
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        
        frames = range(len(spectral_centroid))
        t = librosa.frames_to_time(frames, sr=sample_rate)
        
        plt.subplot(2, 1, 1)
        plt.plot(t, spectral_centroid)
        plt.title('Spectral Centroid')
        plt.ylabel('Hz')
        
        plt.subplot(2, 1, 2)
        plt.plot(t, spectral_rolloff)
        plt.title('Spectral Rolloff')
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        
        plt.tight_layout()
        visualizations['spectral_features'] = self._plot_to_base64()
        
        return visualizations
    
    def _plot_to_base64(self) -> str:
        """Convert current matplotlib plot to base64 string"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"