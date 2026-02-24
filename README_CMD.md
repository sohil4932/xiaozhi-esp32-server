# Command to buld web
docker buildx build --platform linux/amd64 \
  -f Dockerfile-web \   
  -t sohil4932/nokotoy-web:realtime \   
  --push \
  .

# Command to build server image
docker buildx build --platform linux/amd64 \
  -f Dockerfile-server \
  -t sohil4932/nokotoy-server:realtime \
  --push \
  .

# Command to build base image
docker buildx build --platform linux/amd64 \
  -f Dockerfile-server-base \
  -t sohil4932/nokotoy-server:base \
  --push \
  .

# Docker Contaienr access
docker exec -it xiaozhi-esp32-server /bin/bash