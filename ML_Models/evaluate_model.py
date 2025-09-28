import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Load processor & model
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MelodyMachine/Deepfake-audio-detection-V2")
model = AutoModelForAudioClassification.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")

def detect_deepfake(audio_file):
    # Load waveform & resample to 16kHz (required by model)
    waveform, sr = torchaudio.load(audio_file)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    # Preprocess
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted = torch.argmax(scores, dim=-1).item()

    # Map to label
    labels = model.config.id2label
    return {labels[i]: float(scores[0][i]) for i in range(len(labels))}

# Example
result = detect_deepfake("data/Dummy Test Dataset/testing/real/file3.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav")
result2 = detect_deepfake("data/Dummy Test Dataset/testing/fake/file2.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav")
print(result)
print(result2)

