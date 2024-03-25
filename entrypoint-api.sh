#!/bin/bash

source venv/bin/activate 
cd ZEGGS/
uvicorn "main_fastapi:app" --host 0.0.0.0 --port 6000
