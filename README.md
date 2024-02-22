
# Closed Caption Project

 1. Install whisper
 2. Sentence recognition

## Install whisper
    !pip install git+https://github.com/openai/whisper.git
    !sudo apt update && sudo apt install ffmpeg

## Sentence recognition
    !whisper "audio_name.wav" --model medium.en --word_timestamps True --highlight_words True

