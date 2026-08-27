PYTHON ?= python
.PHONY: test train lint smoke
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src tests
train:
	PYTHONPATH=src $(PYTHON) -m train.train --config configs/default.yaml
smoke:
	PYTHONPATH=src $(PYTHON) -m train.train --config configs/toy.yaml
