import numpy as np
import librosa


class AudioAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file

    def get_avg_frequency(self, start_time, end_time, n_fft=512):
        y, sr = librosa.load(self.audio_file, sr=None)

        # Calculate the sample indices for the start and end times
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract the segment within the specified time range
        y_segment = y[start_sample:end_sample]

        stft = np.abs(librosa.stft(y_segment))

        frequencies = librosa.fft_frequencies(sr=sr)

        avg_frequency = np.mean(
            np.sum(stft * frequencies[:, np.newaxis], axis=0) / np.sum(stft, axis=0)
        )

        return avg_frequency

    def get_max_frequency(self, start_time, end_time):
        # Load the audio file
        y, sr = librosa.load(
            self.audio_file, sr=None, offset=start_time, duration=end_time - start_time
        )

        # Extract pitch using librosa's pitch tracking method (e.g., librosa.pyin or librosa.yin)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

        # Filter and find the maximum pitch
        max_pitch = np.max(
            pitches[pitches > 0]
        )  # Assuming pitches is an array where non-zero values represent actual pitches

        return max_pitch

    def get_top10_frequency(self, start_time, end_time, n_fft=512):
        # Load the audio file
        y, sr = librosa.load(self.audio_file, sr=None)

        # Calculate the number of samples for the start and end times
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract the segment of interest
        y_segment = y[start_sample:end_sample]

        # Calculate the short-time Fourier transform (STFT)
        stft = np.abs(librosa.stft(y_segment))

        # Convert to frequencies
        frequencies = librosa.fft_frequencies(sr=sr)

        # Calculate frequency magnitude for each time frame
        freq_magnitudes = np.sum(stft * frequencies[:, np.newaxis], axis=0) / np.sum(
            stft, axis=0
        )

        # Find the top 10 frequencies
        top10_frequencies = np.sort(freq_magnitudes)[-10:]

        return top10_frequencies
