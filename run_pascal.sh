#!/bin/bash

neptune enqueue src/model4.py --config cfg.yaml --storage-url /mnt/ml-team/satellites/max.sokolowski --paths-to-dump src --docker-image ml-docker-repo.deepsense.codilime.com/deepsense/neptune/base-docker:1.4.4-cuda80-tf12 --requirements "pascal"
