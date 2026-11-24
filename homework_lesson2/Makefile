install:
	poetry install
	poetry run pre-commit install

repro:
	poetry run dvc repro

clean:
	rm -rf data/processed models metrics reports
