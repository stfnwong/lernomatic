#!/bin/bash
# RUN TESTS

pytest -vv test/test_inferrer.py
pytest -svv test/test_mnist_trainer.py
pytest -svv test/test_lr_finder.py
pytest -vv test/test_model.py
pytest -vsv test/test_resnet_trainer.py
pytest -vsv test/test_scheduler.py
pytest -vsv test/test_trainer.py
