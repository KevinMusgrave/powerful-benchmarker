black . --exclude notebooks
isort . --profile black --skip-glob notebooks
nbqa black notebooks
nbqa isort notebooks --profile black