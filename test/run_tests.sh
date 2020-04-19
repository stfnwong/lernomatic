#!/bin/bash
# RUN TESTS

pytest -vv test/test_inferrer.py
pytest -vv test/test_mnist_trainer.py
pytest -vv test/test_lr_finder.py
pytest -vv test/test_model.py
pytest -vv test/test_resnet_trainer.py
pytest -vv test/test_scheduler.py
pytest -vv test/test_trainer.py
