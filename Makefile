build:
	docker build  -t whisperx:latest .

run: build
	docker run --rm -it --runtime=nvidia --gpus all --env HF_TOKEN=$$HF_TOKEN -v ./:/app/ whisperx:latest /app/audio/tukevasti-ilmassa-3min.mp3 --model large-v3 --device cuda --compute_type float16 --language fi --batch_size 16 --diarize --output_dir /app/out

run-test-script: build
	docker run --rm -it --runtime=nvidia --gpus all --env HF_TOKEN=$$HF_TOKEN -v ./:/app/ --entrypoint /opt/whisperx-venv/bin/python whisperx:latest test-cuda-or-cpu.py

run-bash: build
	docker run --rm -it --runtime=nvidia --gpus all --env HF_TOKEN=$$HF_TOKEN -v ./:/app/ --entrypoint /bin/bash whisperx:latest

test: build
	docker run --rm -it --runtime=nvidia --gpus all --env HF_TOKEN=$$HF_TOKEN --env S3_INPUT_BUCKET=$$S3_INPUT_BUCKET --env S3_OUTPUT_BUCKET=$$S3_OUTPUT_BUCKET-x --env AWS_ACCESS_KEY_ID=$$AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY=$$AWS_SECRET_ACCESS_KEY --env AWS_SESSION_TOKEN=$$AWS_SESSION_TOKEN --env AWS_REGION=eu-west-1 -v ./.container:/root/ -v /tmp/whisperx:/tmp --read-only whisperx:latest

