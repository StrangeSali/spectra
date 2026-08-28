run_main:
	AUDIO_SOURCE=mic python spectra/main.py

FILE ?= clean_up/raw_data/test_tone.wav
run_file:
	AUDIO_SOURCE=file AUDIO_FILE=$(FILE) python spectra/main.py
