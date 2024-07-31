import os
import re
import json
import numpy as np
import soundfile as sf
import nltk
from nltk.corpus import cmudict
import parselmouth
from pydub import AudioSegment
import librosa

# Ensure nltk resources are downloaded
nltk.download("cmudict")

# Load CMU Pronouncing Dictionary
pronouncing_dict = cmudict.dict()

# Dictionary of word-to-number mappings
word_to_number = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


# Helper function to format time for VTT files
def format_time(seconds):
    milliseconds = int(seconds * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(
        hours, minutes, seconds, milliseconds % 1000
    )


# Main class for sentence recognition and VTT generation
class SentenceRecognizer:
    def __init__(self, audio_file, json_file):
        self.audio_file = audio_file
        self.json_file = json_file
        self.text_sentences = []
        self.sentences = []

    def sentence_recognition(self):
        command = f'whisper "{self.audio_file}" --model medium.en --word_timestamps True --highlight_words True'
        os.popen(command).read()

    def collect_data(self):
        with open(self.json_file, "r") as f:
            data = json.load(f)
        for segment in data["segments"]:
            words_data = self._extract_words(segment)
            self.sentences.extend(words_data)
            self.text_sentences.append(segment["text"])

    def _extract_words(self, segment):
        words_in_sentence = []
        all_words = []
        for idx, word_info in enumerate(segment["words"]):
            word = self._format_word(word_info["word"].strip())
            start = word_info["start"]
            end = word_info["end"]

            if start == end:
                end += 0.01  # Ensure a valid time range

            amp = self._calculate_average_amplitude(start, end)
            syllable_count = self._get_syllable_count(word)

            if self._check_space(word):
                words_list = word.split()
                time_avg = (end - start) / len(words_list)

                for sub_word in words_list:
                    sub_word = self._replace_words_with_numbers(sub_word)
                    sub_start = start
                    sub_end = start + time_avg
                    sub_amp = self._calculate_average_amplitude(sub_start, sub_end)
                    word_tuple = (sub_word, sub_start, sub_end, sub_amp)
                    words_in_sentence.append(word_tuple)
                    start += time_avg
            else:
                word_tuple = (word, start, end, amp)
                words_in_sentence.append(word_tuple)

            if (
                self._contains_punctuation(word_info["word"])
                or idx == len(segment["words"]) - 1
            ):
                all_words.append(words_in_sentence)
                words_in_sentence = []

        if words_in_sentence:
            all_words.append(words_in_sentence)

        return all_words

    def _calculate_average_amplitude(self, start_time, end_time):
        try:
            audio_data, sample_rate = sf.read(self.audio_file)
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio_segment = audio_data[start_sample:end_sample]
            amplitude = np.abs(audio_segment)
            return np.mean(amplitude)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def _get_min_max_amplitudes(self, audio_file_path, start_time, end_time):
        try:
            # Load the audio file and get sample rate
            audio_data, sample_rate = sf.read(audio_file_path)

            # Convert start and end times to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Extract the segment from the audio data
            audio_segment = audio_data[start_sample:end_sample]

            # Find the minimum and maximum absolute amplitudes in the segment
            min_amplitude = np.min(audio_segment)  # The most negative amplitude
            max_amplitude = np.max(audio_segment)  # The most positive amplitude

            return min_amplitude, max_amplitude

        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None

    def _get_pitch_avg(self, start_time, end_time, n_fft=512):

        y, sr = librosa.load(self.audio_file, sr=None)
        
        # Calculate the sample indices for the start and end times
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract the segment within the specified time range
        y_segment = y[start_sample:end_sample]
        
        # Set n_fft to the next power of 2 that is less than or equal to the segment length
        n_fft = 2**int(np.floor(np.log2(len(y_segment))))
        
        # Compute the pitch (fundamental frequency)
        pitches, magnitudes = librosa.core.piptrack(y=y_segment, sr=sr, n_fft=n_fft)
        
        # Select the pitches for each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Only consider positive pitch values
                pitch_values.append(pitch)
    
        # Calculate the average pitch
        avg_pitch = np.mean(pitch_values) if pitch_values else 0
        
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
        n_fft = 2**int(np.floor(np.log2(len(y_segment))))
        
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
            n_fft = 2**int(np.floor(np.log2(len(y_chunk))))
            
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

        

    def _format_word(self, word):
        replacements = {
            "!": "",
            ".": "",
            ",": "",
            "?": "",
            "-": "",
            "0": "zero ",
            "1": "one ",
            "2": "two ",
            "3": "three ",
            "4": "four ",
            "5": "five ",
            "6": "six ",
            "7": "seven ",
            "8": "eight ",
            "9": "nine ",
        }
        for char, replacement in replacements.items():
            word = word.replace(char, replacement)
        return word.strip()

    def _check_space(self, word):
        return " " in word

    def _contains_punctuation(self, word):
        punctuation_marks = [".", "?", "!"]
        return any(char in punctuation_marks for char in word)

    def _replace_words_with_numbers(self, word):
        for char, replacement in word_to_number.items():
            word = word.replace(char, replacement)
        return word.strip()

    def _get_syllable_count(self, word):
        if word.lower() in pronouncing_dict:
            pronunciation = pronouncing_dict[word.lower()][0]
            syllable_count = sum(
                1 for phoneme in pronunciation if phoneme[-1].isdigit()
            )
            return syllable_count
        return None

    def split_text(self):
        result = []
        pattern = r"([.?!])"
        for text in self.text_sentences:
            split_texts = re.split(pattern, text)
            sentences = [
                "".join(pair)
                for pair in zip(split_texts[::2], split_texts[1::2])
                if pair[0].strip()
            ]
            result.extend(sentences)
        return result

    def _get_stress_value(self, sentence_amplitude, word_amplitude):
        if word_amplitude > (sentence_amplitude * 2):
            return 2  # High pitch
        elif word_amplitude >= sentence_amplitude:
            return 1  # Primary stress
        else:
            return 0  # No stress

    def format_texts(self, texts, stress_lists):
        formatted_texts = []
        for text, stress_list in zip(texts, stress_lists):
            formatted_text = self._apply_stress_formatting(text, stress_list)
            formatted_texts.append(formatted_text)
        return formatted_texts

    def _apply_stress_formatting(self, texts, stress_list):
        # print(stress_list)
        formatted_result = []

        # Index for keeping track of position in the text
        text_index = 0

        # Process each (word, stress) tuple in the stress list
        for word, stress in stress_list:
            # Find the start index of this word in the text
            start = texts.find(word, text_index)
            end = start + len(word)

            # Add the text leading up to the word (if there are punctuation/symbols, preserve them)
            formatted_result.append(texts[text_index:start])

            # Format the word based on the stress value
            if stress == 2:
                formatted_word = f"<b>{word.upper()}</b>"
            elif stress == 1:
                formatted_word = f"<u>{word}</u>"
            else:
                formatted_word = word

            # Add the formatted word to the result
            formatted_result.append(formatted_word)

            # Update text_index to the end of the current word
            text_index = end

        # Add any remaining text after the last word
        formatted_result.append(texts[text_index:])
        formatted_html = "".join(formatted_result)
        # Join all parts to create the final formatted string
        return formatted_html.strip()

    def combine_texts_with_timing(self, formatted_texts, sentence_data):
        formatted_sentences = []
        for formatted_text, sentence in zip(formatted_texts, sentence_data):
            start_time = sentence[0][1]
            end_time = sentence[-1][2]
            formatted_sentences.append((start_time, end_time, formatted_text))
        return formatted_sentences

    def write_vtt(self, vtt_filename, formatted_sentences):
        with open(vtt_filename, "w") as f:
            f.write("WEBVTT\n\n")
            for index, (start, end, text) in enumerate(formatted_sentences, start=1):
                f.write(f"{index}\n")
                f.write(f"{format_time(start)} --> {format_time(end)}\n")
                f.write(f"{text}\n\n")

    def generate_vtt(self, vtt_filename):
        self.collect_data()
        texts = self.split_text()
        stress_lists = self._calculate_stress()
        print(stress_lists)
        formatted_texts = self.format_texts(texts, stress_lists)
        formatted_sentences = self.combine_texts_with_timing(
            formatted_texts, self.sentences
        )
        self.write_vtt(vtt_filename, formatted_sentences)

    def _calculate_stress(self):
        stress_lists = []
        for sentence in self.sentences:
            sentence_amplitude = self._calculate_average_amplitude(sentence[0][1], sentence[-1][2])
            pitch_avg_sentence = self._get_pitch_avg(sentence[0][1], sentence[-1][2])
            pitch_max_sentence = self._get_pitch_max(sentence[0][1], sentence[-1][2])

            print(f"AVG AMPLITUDE: {sentence_amplitude}")
            print(f"AVG PITCH: {pitch_avg_sentence}")
            print(f"MAX PITCH: {pitch_max_sentence}")

            word_stress_list = []
            for word_info in sentence:
                word, start_time, end_time, amp = word_info
                syllable_count = self._get_syllable_count(word)

                if syllable_count and syllable_count > 1:
                    syllables = self._split_word_into_syllables(word, syllable_count)
                    # Calculate the average duration per syllable
                    time_avg = (end_time - start_time) / syllable_count

                    sub_start = start_time
                    for i, sub_word in enumerate(syllables):
                        # If it's the last subword, use the original end time
                        if i == len(syllables) - 1:
                            sub_end = end_time
                        else:
                            # Calculate sub_end based on the average duration per syllable
                            sub_end = sub_start + time_avg
                        sub_amp = self._calculate_average_amplitude(sub_start, sub_end)
                        stress_value = self._get_stress_value(
                            sentence_amplitude, sub_amp
                        )

                        pitch_avg = self._get_pitch_avg(sub_start, sub_end)
                        pitch_max = self._get_pitch_max(sub_start, sub_end)
                        pitch_top10_avg = self._get_pitch_top10_avg(sub_start, sub_end)

                        # Print for debugging purposes
                        print(
                            f"  {sub_word}, {sub_start}, {sub_end}, {sub_amp}, {stress_value}"
                        )
                        print(f"        pitch avg: {pitch_avg}")
                        print(f"        pitch max: {pitch_max}")
                        print(f"        pitch top10 avg: {pitch_top10_avg}")
                        # Append to word_stress_list
                        word_stress_list.append((sub_word, stress_value))
                        # Update sub_start to the next time interval
                        sub_start = sub_end

                else:
                    stress_value = self._get_stress_value(sentence_amplitude, amp)
                    word_stress_list.append((word, stress_value))
                    # Fetch min_amplitude, max_amplitude from _get_min_max_amplitudes
                    min_amplitude, max_amplitude = self._get_min_max_amplitudes(
                        self.audio_file, start_time, end_time
                    )
                    # Print for debugging purposes
                    print(
                        f"  {word}, {start_time}, {end_time}, {amp}, {stress_value}"
                    )
                    pitch_avg = self._get_pitch_avg(start_time, end_time)
                    pitch_max = self._get_pitch_max(start_time, end_time)
                    pitch_top10_avg = self._get_pitch_top10_avg(start_time, end_time)

                    print(f"        pitch avg: {pitch_avg}")
                    print(f"        pitch max: {pitch_max}")
                    print(f"        pitch top10 avg: {pitch_top10_avg}")
            print("")
            stress_lists.append(word_stress_list)
        return stress_lists


    def _split_word_into_syllables(self, word, num_syllables):
        vowels = "aeiouy"
        syllables = []
        current_syllable = ""
        for letter in word:
            if letter.lower() in vowels:
                if current_syllable:
                    syllables.append(current_syllable)
                    current_syllable = ""
            current_syllable += letter
        if current_syllable:
            syllables.append(current_syllable)

        if len(syllables) != num_syllables:
            syllables = []
            syllable_length = len(word) // num_syllables
            remaining_chars = len(word) % num_syllables
            start_index = 0
            for i in range(num_syllables):
                end_index = start_index + syllable_length
                if i < remaining_chars:
                    end_index += 1
                syllables.append(word[start_index:end_index])
                start_index = end_index
        return syllables


"""vtt_file_name = "stress_closed_caption.vtt"
audio_file = "vdo23.wav"
json_file = "vdo23.json"

recognizer = SentenceRecognizer(audio_file, json_file)
recognizer.generate_vtt(vtt_file_name)
"""