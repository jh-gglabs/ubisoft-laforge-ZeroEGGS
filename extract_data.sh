#!/bin/bash

cat ./data/Zeggs_data.z01 ./data/Zeggs_data.z02 ./data/Zeggs_data.zip > ./data/Zeggs_data_full.zip
unzip ./data/Zeggs_data_full.zip -d ./data/
