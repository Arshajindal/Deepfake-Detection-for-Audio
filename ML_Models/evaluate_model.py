import os
import torch
import torchaudio
import pandas as pd
from transformers import AutoProcessor, AutoModelForAudioClassification

# Load processor & model once
processor = AutoProcessor.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")
model = AutoModelForAudioClassification.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")

def detect_deepfake(audio_file):
    # Load waveform
    waveform, sr = torchaudio.load(audio_file)

    # Resample to 16kHz (model requirement)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    # Preprocess
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Map to labels
    labels = model.config.id2label
    return {labels[i]: float(scores[0][i]) for i in range(len(labels))}

def evaluate_folder(folder_path):
    results = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".wav"):  # Only process wav files
            fpath = os.path.join(folder_path, fname)
            scores = detect_deepfake(fpath)
            results.append({
                "file": fname,
                "real_score": scores.get("Real", 0.0),
                "fake_score": scores.get("Fake", 0.0),
                "predicted_label": "Real" if scores.get("Real", 0.0) > scores.get("Fake", 0.0) else "Fake"
            })
    return pd.DataFrame(results)

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    folder = "../data/Dummy Test Dataset/testing/fake"
    df = evaluate_folder(folder)
    print(df)
    # Save to CSV if needed
    df.to_csv("results_fake_folder.csv", index=False)
