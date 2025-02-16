# MacWhisperer

## Overview
This is intended to be simple example for speech to text, running locally on mac. This example uses [insanely-fast-whisper project](https://github.com/Vaibhavs10/insanely-fast-whisper), which includes multiple clever optimizations and works with mac mps gpu.

## Prerequisites

- macOS with Apple Silicon (M1 or later)
- Python 3.9 or later
- (Optional) Setup a Virtual Environment

    ```python3 -m venv whisper_env```

    ```source whisper_env/bin/activate```

- Homebrew (for package management)
- ```brew install ffmpeg``` used for transcoding input files to the necessary format 
- ```pip install insanely-fast-whisper```


## Step 1: Transcode input into WAV format
```ffmpeg -i input.mp4 -acodec pcm_s16le -ar 16000 output.wav```
- -i input.mp4: Specifies the input MP4 file.
- -acodec pcm_s16le: Sets the audio codec to PCM signed 16-bit little-endian, a common format for WAV files.
- -ar 16000: Sets the audio sample rate to 16 kHz, which is often required by transcription models.
- output.wav: Specifies the name of the output WAV file.

## Step 2: insanely-fast-whisper
```insanely-fast-whisper  --language en --num-speakers 1 --device-id mps  --file-name 2025s-chris-anderson-002-7ad7e666-d081-41ca-9c29-1200k.wav --transcript-path 2025s-chris-anderson-002-7ad7e666-d081-41ca-9c29-1200k.json```

## Wrapper script to combine steps
```transcribe_mp4.sh <filename>```
(Assumed MP4 format)

## Sample performance test results
Example of transcription performance using a few public domain videos as input

| Sample | Sample Duration | Transcription Time (M3, 16GB) | Comment|
| --------------- | --------------- | --------------- | --------------- |
| [sounds-of-mars-one-small-step-earth](https://science.nasa.gov/resource/sounds-of-mars-one-small-step/) | 0:10 | 0:02 | 
| [2024s-victor-riparbelli](https://www.ted.com/talks/quick-list) | 16:24 | 04:37 | First sentence, near loud Ted intro music, missing
[2025s-chris-anderson](https://www.ted.com/talks/quick-list) |57:09 | N.A. | OOM Kill @ 0:06:22

## Future work
Learn how to
- set attention mask
- use Flash Attention 2
