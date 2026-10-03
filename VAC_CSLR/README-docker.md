# VAC_CSLR no Docker

## Pré-requisitos:

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
- NVIDIA Driver na versão >= 450.80.02 (Linux) ou >= 452.39 (Windows), pois o VAC_CSLR usa CUDA 11.x (verificar possíveis mudanças na versão do driver [aqui](https://docs.nvidia.com/deploy/cuda-compatibility/))

## > Criação da imagem Docker

Neste caso, uma imagem Docker é criada para prover todas as dependências necessárias pelo VAC_CSLR. Para criar essa imagem, o seguinte comando deve ser executado a partir deste diretório:

```bash
docker build -t captar-vac-cslr .
```

O processo de criação da imagem normalmente é demorado (10 min ~ 15 min) e a imagem final é relativamente grande (~7.8 GB).

### > Criação do contêiner

Após a criação da imagem, ela deve ser utilizada para executar a aplicação. Para isso, um contêiner deve ser criado a partir deste diretório usando-se o comando:

```bash
docker run -it --runtime=nvidia --gpus all -v "$PWD":/VAC_CSLR/ captar-vac-cslr
```

Se houver soft links nesta pasta que são usados durante a execução da aplicação, seus diretórios destino devem ser passados como volumes na criação do contêiner também. Por exemplo, se o diretório `/mnt/washingtonramos/` é o destino de um soft link, então a execução do comando acima deve ser alterado para:

```bash
docker run -it --runtime=nvidia --gpus all -v "$PWD":/VAC_CSLR/ -v /mnt/washingtonramos/:/mnt/washingtonramos/ captar-vac-cslr
```
