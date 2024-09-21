import numpy as np
import librosa


class AudioAnalyzer:
    def __init__(self, audio_file):
        self.audio_file = audio_file

    def _get_pitch_avg(self, start_time, end_time, n_fft=512):

        y, sr = librosa.load(self.audio_file, sr=None)

        # Calculate the sample indices for the start and end times
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract the segment within the specified time range
        y_segment = y[start_sample:end_sample]

        # Set n_fft to the next power of 2 that is less than or equal to the segment length
        n_fft = 2 ** int(np.floor(np.log2(len(y_segment))))

        # Compute the pitch (fundamental frequency)
        pitches, magnitudes = librosa.core.piptrack(y=y_segment, sr=sr, n_fft=n_fft)

        # Select the pitches for each frame
        pitch_values = []
        
        index = 0
        # print(pitches.size)
        # print(pitches.ndim)
        avg_pitch = np.mean(pitches)
        # print(avg_pitch)
        
        # for t in range(pitches.size):
        #     pitch = pitches[index, t]
        #     index = magnitudes[:, t].argmax()
        #     # if pitch > 0:  # Only consider positive pitch values
        #     pitch_values.append(pitch)
            
        # print(pitch_values)
        # for t in range(pitches.shape[1]):
        #     index = magnitudes[:, t].argmax()
        #     pitch = pitches[index, t]
        #     print("index: " , index)
        #     print("pitch: " , pitch)
        #     print("t: " , t)
        #     print("")
            
        #     # if pitch > 0:  # Only consider positive pitch values
        #     pitch_values.append(pitch)
            
        # print(pitch_values)
        
        # Calculate the average pitch
        # avg_pitch = np.mean(pitch_values) if pitch_values else 0

        return avg_pitch

    def _get_pitch_max(self, start_time, end_time):
        # Load the audio file
        y, sr = librosa.load(self.audio_file, sr=None)

        # Calculate the sample indices for the start and end times
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract the segment within the specified time range
        y_segment = y[start_sample:end_sample]

        # Set n_fft to the next power of 2 that is less than or equal to the segment length
        n_fft = 2 ** int(np.floor(np.log2(len(y_segment))))

        # Compute the pitch (fundamental frequency)
        pitches, magnitudes = librosa.core.piptrack(y=y_segment, sr=sr, n_fft=n_fft)

        # Select the pitches for each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Only consider positive pitch values
                pitch_values.append(pitch)

        # Calculate the maximum pitch
        max_pitch = np.max(pitch_values) if pitch_values else 0

        return max_pitch

    def _get_pitch_top10_avg(self, start_time, end_time, chunk_size=0.1):

        y, sr = librosa.load(self.audio_file, sr=None)

        # Calculate the sample indices for the start and end times
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract the segment within the specified time range
        y_segment = y[start_sample:end_sample]

        # Calculate the number of chunks
        chunk_length = int(chunk_size * sr)
        num_chunks = len(y_segment) // chunk_length

        avg_pitches = []

        # Process each chunk
        for i in range(num_chunks):
            chunk_start = i * chunk_length
            chunk_end = chunk_start + chunk_length
            y_chunk = y_segment[chunk_start:chunk_end]

            # Set n_fft to the next power of 2 that is less than or equal to the chunk length
            n_fft = 2 ** int(np.floor(np.log2(len(y_chunk))))

            # Compute the pitch (fundamental frequency)
            pitches, magnitudes = librosa.core.piptrack(y=y_chunk, sr=sr, n_fft=n_fft)

            # Select the pitches for each frame
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Only consider positive pitch values
                    pitch_values.append(pitch)

            # Calculate the average pitch for the chunk
            avg_pitch = np.mean(pitch_values) if pitch_values else 0
            avg_pitches.append(avg_pitch)

        # Get the top 10 average pitches
        top_10_avg_pitches = sorted(avg_pitches, reverse=True)[:10]

        # Calculate the average of the top 10 pitches
        top_10_avg_pitch = np.mean(top_10_avg_pitches)

        return top_10_avg_pitch
