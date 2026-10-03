docker run -it --gpus "device=0" --shm-size 64G \
  -v "$PWD":/VAC_CSLR/ \
  -v /srv/projects2/captarlibras_finep/sign-to-text/dataset/:/srv/projects2/captarlibras_finep/sign-to-text/dataset/ \
  -v /draft-hdd-projects/captarlibras_finep/cauamagalhaes/datasets/:/draft-hdd-projects/captarlibras_finep/cauamagalhaes/datasets/ \
  -v /srv/projects2/captarlibras_finep/output_gravacoes_conv/:/srv/projects2/captarlibras_finep/output_gravacoes_conv/ \
  -v /draft-ssd-projects/captarlibras_finep/gabrielbezerra/datasets/:/draft-ssd-projects/captarlibras_finep/gabrielbezerra/datasets/ \
  -v /draft-hdd-projects/captarlibras_finep/gabrielbezerra/datasets/:/draft-hdd-projects/captarlibras_finep/gabrielbezerra/datasets/ \
  -v /srv/projects2/captarlibras_finep/sign-to-text/dataset/captar-libras_17-05/raw_data/:/srv/projects2/captarlibras_finep/sign-to-text/dataset/captar-libras_17-05/raw_data/ \
  -v /draft-ssd-projects/captarlibras_finep/datasets/:/draft-ssd-projects/captarlibras_finep/datasets/ \
  -v /srv/projects2/captarlibras_finep/dataset_captarlibras/annotation-data/ELAN-output-frontal-cam/:/srv/projects2/captarlibras_finep/dataset_captarlibras/annotation-data/ELAN-output-frontal-cam/ \
  -v /draft-ssd-projects/captarlibras_finep/arigsf:/draft-ssd-projects/captarlibras_finep/arigsf \
  -v /srv/:/srv/ \
  captar-vac-cslr
