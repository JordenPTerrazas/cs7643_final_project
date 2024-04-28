"""There are 300 real audio clips with no ground truth, we will pick 16 at
random to test the model. We will use the same preprocessing as the synthetic"""
import os
import random
import torchaudio

if __name__ == '__main__':
    real_dir = "data/datasets/test_set/real_recordings"
    real_audio_files = [os.path.join(real_dir, file) for file in os.listdir("data/datasets/test_set/real_recordings")]
    
    random.shuffle(real_audio_files)

    # Grab only 10 sec audio files, so we dont have a padding nightmare
    counter = 0
    subset = []
    for audio_file in real_audio_files:
        if counter == 16:
            break
        wave, sample_rate = torchaudio.load(audio_file)
        if wave.shape[1] == 160000:
            counter += 1
            subset.append(audio_file)

    os.makedirs("data/datasets/test_set/real_recordings_subset", exist_ok=True)

    # Move the files over
    for file in subset:
        os.rename(file, os.path.join("data/datasets/test_set/real_recordings_subset", os.path.basename(file)))
