#!/bin/bash

conda activate cosyvoice && uvicorn FastAPIdemo:app --host 0.0.0.0 --port 8080