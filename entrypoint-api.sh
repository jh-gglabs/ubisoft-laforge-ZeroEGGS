#!/bin/bash

source venv/bin/activate 
cd ZEGGS/
uvicorn "main_fastapi:app"
