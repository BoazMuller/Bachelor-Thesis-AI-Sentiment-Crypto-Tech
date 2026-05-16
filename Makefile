.PHONY: setup test notebook clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -r requirements.txt
	. .venv/bin/activate && python -m pip install -e code

test:
	pytest

notebook:
	jupyter lab

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

