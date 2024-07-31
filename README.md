
# Sentence Recognizer and VTT Generator

This repository contains a Python implementation for recognizing sentences from audio files and generating WebVTT (VTT) files with stress-marked closed captions. The process includes extracting features like amplitude and pitch to determine the stress on syllables and words.

## Features

-   **Sentence Recognition:** Uses OpenAI's Whisper model for recognizing sentences from audio files.
-   **Stress Calculation:** Determines stress levels on syllables and words based on amplitude and pitch analysis.
-   **VTT Generation:** Generates WebVTT files with formatted text to aid in pronunciation learning.

## Installation

### Prerequisites

-   Python 3.6 or higher
-   [Whisper](https://github.com/openai/whisper) model from OpenAI

### Dependencies

You can install the required dependencies using `pip`. It is recommended to use a virtual environment to manage dependencies.

`pip install numpy soundfile nltk parselmouth pydub librosa` 

### Whisper Installation

Follow the instructions from the [Whisper repository](https://github.com/openai/whisper) to install Whisper and its dependencies.
