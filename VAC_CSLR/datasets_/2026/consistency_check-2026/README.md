Tendo a coluna VideoID basta procura em /srv/projects2/captarlibras_finep/dataset_captarlibras/** os videos corespondentes, exemplo:

```
find ./ -type f -name "01-02-2025_09-28-58_p0069*"

./raw-data/face-cam/01-02-2025_09-28-58_p0069_sP18.1_b4_RAW--webcam-CANON_FACE.avi
./raw-data/depth-cam/01-02-2025_09-28-58_p0069_sP18.1_b4_RAW--ZED-DEPTH_FRONT.mkv
./raw-data/superior-cam/01-02-2025_09-28-58_p0069_sP18.1_b4_RAW--webcam-LOGI_TOPO.avi
./raw-data/diagonal-cam/01-02-2025_09-28-58_p0069_sP18.1_b4_RAW--webcam-LOGI_SIDE.avi
./raw-data/frontal-cam/01-02-2025_09-28-58_p0069_sP18.1_b4_RAW--ZED-RGB_FRONT.avi
./annotation-data/ELAN-output-frontal-cam/01-02-2025_09-28-58_p0069_sP18.1_b4_CONV--ZED-RGB.pfsx
./annotation-data/ELAN-output-frontal-cam/01-02-2025_09-28-58_p0069_sP18.1_b4_CONV--ZED-RGB.eaf
```