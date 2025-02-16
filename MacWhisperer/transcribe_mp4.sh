#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_file.mp4>"
  exit 1
fi

# Extract the filename without the extension
input_file="$1"
base_name="${input_file%.*}"

# Define the output filenames
wav_file="${base_name}.wav"
transcript_file="${base_name}.json"

# Convert MP4 to WAV using ffmpeg
echo "Converting $input_file to $wav_file..."
ffmpeg -i "$input_file" -acodec pcm_s16le -ar 16000 "$wav_file"
if [ $? -ne 0 ]; then
  echo "Error during conversion."
  exit 1
fi

# Run insanely-fast-whisper for transcription
echo "Transcribing $wav_file..."
insanely-fast-whisper --language en --num-speakers 1 --device-id mps --file-name "$wav_file" --transcript-path "$transcript_file"
if [ $? -ne 0 ]; then
  echo "Error during transcription."
  exit 1
fi

echo "Transcription completed successfully. Transcript saved to $transcript_file."