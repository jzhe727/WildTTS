#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cosyvoice && uvicorn FastAPIdemo:app --host 0.0.0.0 --port 8080