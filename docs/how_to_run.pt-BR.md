[🇺🇸 English](how_to_run.md) | 🇧🇷 Português

# Como Executar

Este documento descreve o fluxo de execução recomendado para o projeto após a criação
e validação de um diretório de configuração.

O pipeline é executado por meio de um pequeno conjunto de scripts de alto nível.
A execução desses scripts, na ordem correta, produz todos os artefatos do pipeline:

- Conjuntos de dados brutos extraídos
- Conjuntos de dados processados / integrados
- Classificador de BI-RADS para laudos
- Dataset final da coorte para modelagem
- Modelos ajustados e melhores checkpoints
- Métricas de avaliação, gráficos e tabelas de resultados

Todos os scripts dependem dos entrypoints da CLI do pacote, o que exige que o projeto
esteja instalado no ambiente ativo (recomenda-se instalação em modo editável).

---

## 1. Pré-condições

Antes de executar qualquer etapa, certifique-se de que:

1) Existe um diretório de configuração e ele segue o esquema exigido (Ver: docs/configuration_setup.md).

2) O ambiente está instalado e ativo:

```
conda env create -f environment.yml
conda activate hapcancer
```

3) O projeto está instalado em modo editável (a partir da raiz do repositório):

```
pip install -e .
```

4) O defaults mapping está definido para o diretório de configuração utilizado
(followup, embeddings, split, training_experiments, tuning, etc.).

---

## 2. Visão geral: os seis scripts principais

1) run_extract.py:<br>
   Extrai dados brutos das bases de origem de forma particionada (chunked).  
   Requer passagem explícita de --config-dir.

2) run_transform.py:<br>
   Orquestra etapas de transformação e geração de atributos sobre os dados extraídos.  
   Suporta execução parcial via flags de CLI.

3) run_load.py:<br>
   Aplica critérios de elegibilidade e constrói o dataset da coorte.

4) run_tuning.py:<br>
   Executa experimentos de hyperparameter tuning.

5) run_training_best.py:<br>
   Treina o modelo final usando a melhor configuração do tuning.

6) run_eval.py:<br>
   Avalia o modelo final e gera métricas e gráficos.

---

## 3. Modelo de execução: scripts como wrappers de comandos da CLI

Os scripts acima são wrappers em Python para comandos da CLI expostos pelo pacote.
Esses comandos ficam disponíveis após:

```
pip install -e .
```

Em particular, o entrypoint:

```
hapcancer
```

é fornecido pelo módulo de CLI (hapcancer/cli/cli.py) e suporta comandos como:

    hapcancer extract-raw-data [args]
    hapcancer process-birads [args]
    hapcancer preprocess [args]
    hapcancer train-tfidf [args]
    hapcancer merge-sources [args]
    hapcancer generate-dataset [args]
    hapcancer precompute-sequences [args]
    hapcancer tuning [args]
    hapcancer cv-training-best [args]
    hapcancer eval-metrics-best [args]
    hapcancer eval-explain-best [args]

Os scripts wrapper normalmente existem para:

1. Executar múltiplas chamadas à CLI em loops seguros

2. Lidar com variações de origem dos bancos de dados (ex.: HSP vs PSC)

3. Garantir tamanhos de chunk consistentes e reexecutabilidade

4. Reduzir a necessidade de copiar e colar comandos manualmente no terminal

Todos os scripts wrapper exigem que o diretório de configuração seja passado explicitamente
por meio do argumento --config-dir. Caminhos de configuração hardcoded são evitados
intencionalmente para garantir reprodutibilidade e prevenir execuções acidentais
no diretório errado.

---

## 4. Etapa 1: run_extract.py

Objetivo:

&emsp;Extrair dados brutos das bases de origem e armazená-los como arquivos Parquet no diretório apropriado (definido no diretório de configuração).

Configuração necessária:
- etl/paths.yml<br>
- etl/files_and_folders.yml<br>

Padrão típico de execução:
&emsp;A extração é executada por fonte de dados e por banco de dados de origem, geralmente em chunks.

Execução:

```
python run_extract.py --config-dir <config_dir>
```

Exemplo:

```
python run_extract.py --config-dir runs/collection_06012026
```

Observações:

- Como config_dir é explícito, o mesmo script pode ser reutilizado com segurança
em diferentes coleções sem alterações no código.
- Se uma determinada etapa de extração falhar devido a linhas corrompidas, reexecute apenas aquela combinação específica
(raw_data_name, db_origin).
- Se ocorrer interrupção durante a extração, executar novamente fará com que o script retome do ponto
onde parou na última execução.
- Mantenha os tamanhos de chunk estáveis (não altere!) dentro da mesma coleção para evitar misturar
limites de chunk entre reexecuções.

Saídas esperadas:
    
    Arquivos Parquet em:
        >extract_path/<raw_data_folder>/*.parquet

---

## 5. Etapa 2: run_transform.py

Objetivo:
    Orquestrar etapas de transformação sobre os dados brutos extraídos, incluindo limpeza,
    normalização, extração e inferência de BI-RADS, treinamento de modelos auxiliares
    e integração final das fontes de dados.

Entradas:

&emsp;Arquivos Parquet brutos extraídos pelo run_extract.py

Saídas:
- Dados processados contendo BI-RADS extraídos e inferidos a partir dos exames brutos de mamografia.
- Dados processados para cada tipo de dado bruto.
- Laudos de biópsia classificados.
- Modelos TF-IDF ajustados.
- Dados integrados de todas as fontes.

Execução completa:

```
python run_transform.py --config-dir <config_dir>
```

Execução parcial:

```
python run_transform.py --config-dir <config_dir> --step birads-extract
```

Retomar a partir de uma etapa específica (checar o script para lista de etapas):

```
python run_transform.py --config-dir <config_dir> --from preprocess
```

Inspecionar o plano de execução sem executar::

```
python run_transform.py --config-dir <config_dir> --all --dry-run
```

---

## 6. Etapa 3: run_load.py

Objetivo:
&emsp;Aplicar critérios de elegibilidade, construir o dataset da coorte para modelagem e pré-computar agregados históricos de mamografias anteriores para treinamento eficiente.

Entradas:

&emsp;Dataset integrado transformado a partir do run_transform.py

Saídas:

&emsp;`load_path/<dataset_name>/... (datasets finais)`

Execução:

```
python run_load.py --config-dir <config_dir>
```

Arquivos de configuração podem ser alterados diretamente no próprio script, como `strat_cfg` e `gb_sizes`.
`strat_cfg` corresponde ao arquivo de configuração de follow-up.

Por exemplo, se quisermos alterar critérios de inclusão, como o intervalo de idade, `strat_cfg` definirá qual arquivo de configuração deve ser utilizado.
`gb_sizes` refere-se ao espaço de memória dedicado aos vetores pré-computados que representam os agregados dos embeddings de mamografias anteriores. Como esses vetores pré-computados são armazenados em um banco de dados LMDB, o espaço de memória deve ser previamente definido.

---


## 7. Etapa 4: run_tuning.py

Objetivo:
&emsp;Realizar experimentos de hyperparameter tuning utilizando as configurações em model/tuning/*.yml.

Entradas:

- Dataset final da coorte.
- Seleção da configuração de tuning (defaults mapping).

Saídas:

- Logs de tuning.
- Tabelas de resultados do tuning.

Execução:

```
python run_tuning.py --config-dir <config_dir>
```

Os arquivos de configuração que podem ser alterados diretamente no script são: `tuning_cfg`, `split_cfg` e `followup_cfg`.

- `tuning_cfg` refere-se ao arquivo de configuração que define os parâmetros do experimento de tuning (seed, número de trials, pasta de saída, etc.).
- `split_cfg` refere-se ao arquivo de configuração que define os parâmetros de divisão dos dados, como proporções de treino e teste e quais categorias de BI-RADS incluir na coorte (essa divisão deve ser a mesma utilizada posteriormente no treinamento).
- `followup_cfg` refere-se à coorte a ser utilizada no tuning. Deve apontar para um arquivo de configuração já utilizado durante a execução do `run_load.py`.

---

## 8. Etapa 5: run_training_best.py

Objetivo:
&emsp;Treinar um modelo final utilizando a melhor configuração selecionada a partir do tuning.

Entradas:

- Dataset final da coorte
- Melhores hiperparâmetros selecionados (resultado do tuning)

Saídas:

- Checkpoints finais do modelo
- Logs de treinamento
- Snapshot da configuração utilizada (recomendado)

Execução:

```
python run_training_best.py --config-dir <config_dir>
```

Os arquivos de configuração que podem ser alterados diretamente no script são: `tuning_cfg`, `split_cfg` e `followup_cfg`.

- `tuning_cfg` refere-se ao arquivo de configuração utilizado no experimento de tuning correspondente e define a melhor configuração de parâmetros obtida.
- `split_cfg` refere-se ao arquivo de configuração que define os parâmetros de divisão dos dados, como proporções de treino e teste e quais categorias de BI-RADS incluir na coorte (deve ser a mesma divisão utilizada durante o tuning).
- `followup_cfg` refere-se à coorte a ser utilizada no treinamento. Deve apontar para um arquivo de configuração já utilizado durante a execução do `run_load.py`.

---

## 9. Etapa 6: run_eval.py

Objetivo:
&emsp;Avaliar o modelo final no conjunto de teste hold-out e gerar todas as métricas finais, gráficos e tabelas de resultados.

Entradas:
- Checkpoint(s) final(is) do modelo

Saídas:
- Tabelas de métricas
- Gráficos (ROC, PR, calibração, etc.)
- Tabelas de explicabilidade

Execução:

```
python run_eval.py --config-dir <config_dir>
```

Os arquivos de configuração que podem ser alterados diretamente no script são: `tuning_cfg`, `split_cfg` e `followup_cfg`.

- `tuning_cfg` refere-se ao arquivo de configuração utilizado no experimento de tuning correspondente e define a melhor configuração de parâmetros obtida.
- `split_cfg` refere-se ao arquivo de configuração que define os parâmetros de divisão dos dados, como proporções de treino e teste e quais categorias de BI-RADS incluir na coorte (deve ser a mesma divisão utilizada durante o tuning).
- `followup_cfg` refere-se à coorte a ser utilizada no treinamento. Deve apontar para um arquivo de configuração já utilizado durante a execução do `run_load.py`.

---

## 10. Boas práticas

- Não modifique arquivos YAML no meio de uma execução sem registrar a alteração.
- Evite reexecuções parciais a menos que você compreenda as dependências das etapas anteriores.

Uma boa prática é tratar cada diretório de configuração como imutável após a geração dos resultados e criar um novo diretório de configuração para novas coleções. Experimentos de tuning e treinamento podem ser realizados livremente dentro do mesmo diretório de configuração.
