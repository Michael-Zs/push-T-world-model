#!/bin/bash

python -m push_t_jepa.demo --checkpoint artifacts/large/model.pt --output artifacts/demo --seed 7 --steps 240 --visual $@
