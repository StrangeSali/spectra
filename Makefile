run_main:
	AUDIO_SOURCE=mic python spectra/main.py

FILE ?= /Users/anac/Downloads/1-100210-B-36.wav
run_file:
	AUDIO_SOURCE=file AUDIO_FILE=$(FILE) python spectra/main.py
