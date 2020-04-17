#!/bin/bash
# RUN TESTS

pytest -vvx test/test_mnist_trainer.py
pytest -vvx test/test_model.py
pytest -vvx test/test_scheduler.py
pytest -vvx test/test_trainer.py
