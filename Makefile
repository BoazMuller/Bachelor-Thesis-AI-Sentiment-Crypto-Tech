.PHONY: setup notebook clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && python3 -m pip install --upgrade pip
	. .venv/bin/activate && python3 -m pip install -r requirements.txt
	. .venv/bin/activate && python3 -m pip install -e code

notebook:
	jupyter lab

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
