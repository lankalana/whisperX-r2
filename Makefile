build:
	docker build  -t whisperx:latest .

run: build
	docker run --rm -it --runtime=nvidia --gpus all --env HF_TOKEN=$$HF_TOKEN -v ./:/app/ whisperx:latest /app/audio/tukevasti-ilmassa-3min.mp3 --model large-v3 --device cuda --compute_type float16 --language fi --batch_size 16 --diarize --output_dir /app/out

run-test-script: build
	docker run --rm -it --runtime=nvidia --gpus all --env HF_TOKEN=$$HF_TOKEN -v ./:/app/ --entrypoint /opt/whisperx-venv/bin/python whisperx:latest test-cuda-or-cpu.py

run-bash: build
	docker run --rm -it --runtime=nvidia --gpus all --env HF_TOKEN=$$HF_TOKEN -v ./:/app/ --entrypoint /bin/bash whisperx:latest