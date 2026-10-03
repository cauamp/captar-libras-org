docker run -it --gpus "device=0" --shm-size 64G \
  -v "$PWD":/sign2gpt/ \
  -v /srv/:/srv/ \
  captar-sign2gpt
