#!/usr/bin/env bash

conda install pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 -c pytorch
pip install --no-cache-dir -r requirements.txt
