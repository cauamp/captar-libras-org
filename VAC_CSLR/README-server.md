# Inferência usando o websockets

O arquivo `server.py` é um server do websockets para execução de inferência. Para iniciar o server são necessários os parâmetros do modelo que será executado:

```
python3 server.py --load-weights model.pt --config ./configs/config_inference.yaml --phase test
```

Assim que iniciado, é feito o download dos pesos da ResNet18 e são carregados os pesos do arquivo especificado em `model.pt`. Depois de carregado, o modelo fica aguardando as mensagens do websockets para executar a inferência.

O cliente (exemplificado em client.py) deve enviar um json com os seguintes componentes:

message = {
    "input": "input.txt", # path do arquivo de input (string de glosas)
    "output": "output.txt" # path do arquivo onde o output será escrito
}

Quando a mensagem é recebida, o server retorna que recebeu a mensagem e, depois de completa a inferência, retorna um json no formato:

message = {
    "input": "input.txt", # path do arquivo de input (string de glosas)
    "output": "output.txt" # path do arquivo onde o output será escrito
}

Apenas confirmando o path do arquivo de output e conectando ao arquivo de input.