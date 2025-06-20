#! /bin/bash

c2c stop-all

c2c hydra2code --input configs/train_sac.yaml --output project/utils/configs/train_sac.py
