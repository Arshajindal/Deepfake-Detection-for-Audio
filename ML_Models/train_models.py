# predict_audio.py

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio
import argparse
import os

class Wav2VecAIDetector:
    def __init__(self, model_name="Mrkomiljon/voiceGUARD", device=None):
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.labels = ["Real Human Voice", "AI-generated"]

    def predict(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Preprocess
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            labels = self.model.config.id2label
            prediction = labels[predicted_id]
            confidence = float(torch.softmax(logits, dim=-1)[0, predicted_id])

        return {
            "prediction": prediction,
            "confidence": confidence
        }


def main():
    #parser = argparse.ArgumentParser(description="AI vs Human Voice Detection")
    #parser.add_argument("audio_file", type=str, help="Path to audio file (.wav)")
    #args = parser.parse_args()

    detector = Wav2VecAIDetector()
    folder = "data/Dummy Test Dataset/testing/real_2"  # Change to your folder path

    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            path = os.path.join(folder, filename)
            result = detector.predict(path)
            print(f"{filename}: Prediction = {result['prediction']}, Confidence = {result['confidence']*100:.2f}%")

if __name__ == "__main__":
    main()
