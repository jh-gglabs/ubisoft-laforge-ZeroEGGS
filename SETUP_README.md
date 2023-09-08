# How to setup and deploy

## Pytorch Module
* Streaming Data를 Input으로 각 프레임마다의 Body Gesture 정보를 받을 수 있도록 설계되어 있음
```bash
# Docker Build
docker build --no-cache -t zeggs-module -f ./Dockerfile.module .
docker run -d --name zeggs-module-test zeggs-module
```

## FastAPI
* Audio File을 Input으로 전체 프레임의 Body Gesture 정보를 받을 수 있도록 설계되어 있음
```bash
docker build --no-cache -t zeggs-api -f ./Dockerfile.api .
docker run -d --name zeggs-api-test -p 6000:6000 zeggs-api
```
