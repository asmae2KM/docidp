.PHONY: install test lint format dataset eval app docker

install:
	pip install -e ".[dev]"

test:
	pytest --cov=docidp --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

dataset:            ## regenerate the synthetic benchmark (deterministic, seed 42)
	python -m evaluation.generate_dataset --out evaluation/dataset --seed 42

eval:               ## run the live benchmark (needs API keys) and refresh the README results block
	python -m evaluation.run_eval --update-readme

app:
	streamlit run app.py

docker:
	docker build -t docidp . && docker run --rm --env-file .env -p 8501:8501 docidp
