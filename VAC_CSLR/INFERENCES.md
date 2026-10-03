# Documentação dos Scripts de Inferência

## IMPORTANTE

Em todos os scripts, recomenda-se o uso de um **arquivo de configuração** ao invés da especificação dos parâmetros diretamente na linha de comando. Certifique-se de que o arquivo esteja devidamente configurado com os detalhes e caminhos necessários antes de executar o script.

No arquivo de configuração, são definidos fatores cruciais, como:
- **Dataloader:** Deve ser semelhante ao utilizado no treinamento para garantir consistência na performance.
- **Dataset:** O dataset especificao define a versão do gloss_dict que será utilizada na inferencia, essa  deve ser compatível com a versão usada no treinamento, garantindo compatibilidade dos pesos, entre outros fatores essenciais.

---

### `eval_inference.py`

O script `eval_inference.py` realiza inferências nas partições **dev** e **test** de um dataset especificado.

#### Funcionalidades Principais:
- Executa inferências nas partições **dev** e **test**.
- Utiliza um arquivo de configuração para definir o dataset e os parâmetros relevantes.

#### Como Executar:
```bash
python3 eval_inference.py --config ./configs/captar_eval_inference.yaml
```

#### Logs:
Os resultados das inferências são salvos em:
- `<work_dir>/log_dev.txt`
- `<work_dir>/log_test.txt`

---

### `inference_all.py`

Esse script realiza inferência em **todos os vídeos** de uma pasta específica.

#### Funcionalidades:
- Inferências em lotes para vídeos organizados em uma única pasta.
- A pasta de vídeos deve ser especificada dentro do código (linha 204 - **#TODO parametrizar**).

#### Como Executar:
```bash
python3 inference_all.py --config ./configs/captar_inference.yaml
```

---

### `inference_one_video.py`

Esse script realiza inferência em **um único vídeo** ou em uma **pasta de frames** de um vídeo.

#### Funcionalidades:
- Suporta tanto arquivos `.mp4` quanto pastas contendo frames de vídeo.

#### Como Executar:
```bash
python3 inference_one_video.py --config ./configs/captar_inference.yaml --input <caminho para pasta ou .mp4>
```

---

Certifique-se de revisar e ajustar os arquivos de configuração antes da execução, garantindo que as variáveis relevantes estejam corretamente definidas para atender às suas necessidades.
