This data was sourced from Common Voice Delta Segment 20.0 of 12/11/2024. We went through the validated sentences (`validated.tsv`) and picked 10 samples of different speakers with their respective audio files. We converted the downloaded mp3 files to wav using ffmpeg.

Disclaimer: Since the data that Whisper (and most standard STT models) was trained on was mostly English, further finetuning with just a handful of English samples, each from a different speaker, probably won't make an impact in performance.
