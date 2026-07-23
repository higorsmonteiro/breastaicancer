[🇺🇸 English](configuration_setup.md) | 🇧🇷 Português

# Configuração do Sistema

Este projeto utiliza um sistema de configuração baseado em pastas. Um “diretório de configuração”
é uma pasta autocontida que contém um esquema fixo de subpastas e arquivos YAML.
A classe `ConfigInterface` (e todas as classes que herdam dela) depende desse
diretório para carregar configurações, resolver defaults e construir todos os caminhos
de sistema de arquivos utilizados ao longo do pipeline. Esse padrão é utilizado para reduzir
código repetitivo (*boilerplate*) e para centralizar atribuições importantes (e numerosas)
de parâmetros/argumentos.

O sistema de configuração é implementado em:

- hapcancer/config_manager.py
  - ConfigManager (resolve o esquema do diretório e carrega arquivos)
  - ConfigInterface (configuração de alto nível)

Neste documento, explicamos os detalhes de implementação envolvidos.

---

## 1. O que é um “diretório de configuração”

Um diretório de configuração é uma pasta com estrutura conhecida que contém
arquivos YAML organizados por área do pipeline. O exemplo ilustrativo
abaixo segue o esquema atual de um diretório de configuração:

    collection_folder/
      etl/
        paths.yml
        files_and_folders.yml
        fields.yml
        followup/
          age_strat_18_75.yml
          age_strat_40_75.yml
        birads_classifier/
          birads_clf_001.yml
        bmi_model/
          bmi_model_001.yml
        embeddings/
          tfidf_001.yml
          tfidf_002.yml

      model/
        extra.yml
        fields.yml
        splits/
          split_001.yml
          split_002.yml
        training_experiments/
          base.yml
          trial_*.yml
        tuning/
          bce_all_001.yml
          bce_all_002.yml

Esse diretório pode ser trocado (por exemplo, para diferentes experimentos ou coleções),
e o código carregará uma árvore de configuração diferente de acordo.

### 1.1. Descrições

1. **ETL**  
   Coordena os caminhos, nomes de arquivos e parâmetros relacionados à primeira extração
   dos bancos de dados da Hapvida até o dataset final pronto para desenvolvimento de modelo.

   1.1. **paths.yml**  
        Define IDs e caminhos de armazenamento para o ETL. Também define o caminho para o arquivo `.env`
        contendo credenciais de acesso ao banco de dados.<br>

   1.2. **files_and_folders.yml**  
        Define nomes de pastas para cada etapa do ETL e nomes de arquivos de saída.<br>

   1.3. **fields.yml**  
        Campos utilizados durante a análise. Atualmente fixo (e possivelmente desnecessário
        como arquivo de configuração).<br>

   1.4. **followup**  
        Define restrições importantes da coorte.<br>

   1.5. **birads_classifier**  
        Um classificador de BI-RADS (a partir de laudos) é construído durante o ETL.
        Aqui definimos os parâmetros do modelo.<br>

   1.6. **bmi_model**  
        Pode ser removido futuramente. O único necessário é um estimador de IMC
        a partir de sexo e idade.<br>

   1.7. **embeddings**  
        Parâmetros para o modelo de embedding TF-IDF. Outros modelos podem ser incluídos posteriormente.<br>

2. **MODEL**  
   Coordena definições de arquitetura de modelo, procedimentos de treinamento, configuração
   de avaliação e variantes experimentais.

   2.1. **extra.yml**  
        Caminhos globais em nível de modelo e opções de execução (logging, checkpoints, dispositivo).<br>

   2.2. **fields.yml**  
        Define colunas de atributos, alvos, janelas de follow-up e máscaras de elegibilidade.<br>

   2.3. **splits**  
        Define estratégias de divisão de dados e parâmetros relacionados.<br>

   2.4. **training_experiments**  
        Configurações de treinamento (arquitetura e hiperparâmetros).<br>

   2.5. **tuning**  
        Configuração de hyperparameter tuning.<br>

---

### 1.2. High-level access via `ConfigInterface`

O seguinte padrão é utilizado para interação com a pasta de configuração e com os arquivos de configuração:

```python
from hapcancer.config_manager import ConfigInterface

config_dir = "path/to/config_folder"
config_defaults = {
    "birads_classifier": "birads_clf_001.yml",
    "embeddings": "tfidf_001.yml",
    "split": "split_001.yml",
    "followup": "age_strat_18_75.yml",
    "bmi_model": "bmi_model_001.yml",
    "tuning": "bce_all_001.yml",
    "training_experiments": None
}

config = ConfigInterface(config_dir, config_defaults)
```

Quando `ConfigInterface` é instanciada, ela processa diversos valores definidos nos arquivos de configuração carregados. A maioria das classes definidas no software herda diretamente de `ConfigInterface`, compartilhando o mesmo padrão de inicialização.

```python
from hapcancer.etl.extract.extractor import Extractor

with Extractor(config_dir, config_defaults) as extractor:
    extractor.fetch_mammograms_paginated(
        db_origin="hsp",
        start_year=2016,
        timer=0.5,
        chunk_size=200_000,
        verbose=True,
        max_retries=5,
    )
```

Dentro da mesma pasta de configuração, é possível testar diferentes parâmetros para diferentes partes do pipeline. Por exemplo, ao utilizar um modelo TF-IDF para representar laudos de mamografia, múltiplas configurações de embedding podem existir em `etl/embeddings/` (por exemplo, `tfidf_001.yml`, `tfidf_002.yml`). O dicionário config_defaults define os componentes intercambiáveis do pipeline. Para alterar o modelo de embedding utilizado por padrão, pode-se definir:

```python
config_defaults["embeddings"] = "tfidf_002.yml"
```

e reinicializar a classe.

Durante execuções padrão em produção, esse nível de interação não é necessário. Esses detalhes são incluídos caso seja necessário desenvolvimento adicional ou experimentação. `ConfigInterface` é construída sobre o `ConfigManager`.

## 2. Como os arquivos de configuração são carregados

Na inicialização, `ConfigManager` percorre recursivamente o diretório de configuração
e constrói uma árvore em memória com o mesmo formato da estrutura de pastas.

Comportamentos principais:

- Cada pasta se torna uma chave de dicionário aninhada.
- Cada arquivo é carregado e armazenado sob uma chave igual ao seu nome sem extensão.
  Exemplo:
      etl/embeddings/tfidf_001.yml
  é acessível como:
      ["etl", "embeddings", "tfidf_001"]

Observações:

- Se dois arquivos na mesma pasta tiverem o mesmo stem (por exemplo, a.yml e a.json),
  um sobrescreverá o outro. Evite colisões de stem.
- O carregador de arquivos é `hapcancer.etl.utils.load_config_file(filepath)`.

### 2.1. Padrão de acesso

A configuração é acessada por uma lista de chaves:

    cfg_manager.get(["etl", "paths"])
    cfg_manager.get(["etl", "embeddings", "tfidf_001"])
    cfg_manager.get(["model", "splits", "split_001"])

Essa API baseada em caminhos é intencionalmente explícita e espelha a hierarquia de pastas.

### 2.2. Defaults: selecionando “qual configuração usar”

Muitas pastas contêm múltiplos arquivos YAML candidatos (por exemplo, múltiplas configurações de split,
múltiplas configurações de embedding, múltiplas definições de follow-up). O sistema seleciona
qual utilizar por meio de um “mapeamento de defaults”.

Os defaults são definidos chamando:

    cfg_manager.set_defaults(config_defaults)

O mapeamento de defaults aponta para nomes de arquivos (strings). Internamente, o gerenciador
converte cada nome para seu stem para busca na árvore.

Exemplo:

    config_defaults = {
      "followup": "age_strat_30_75.yml",
      "embeddings": "tfidf_003.yml",
      "birads_classifier": "birads_clf_001.yml",
      "bmi_model": "bmi_model_001.yml",
      "split": "split_001.yml",
      "training_experiments": "base.yml",
      "tuning": "bce_all_001.yml",
      "eval": null
    }

Após validação, esses valores são armazenados como stems:

    followup = "age_strat_30_75"
    embeddings = "tfidf_003"
    split = "split_001"
    ...

Os defaults são validados pelo schema Pydantic:

    hapcancer.schemas.validation_models.ConfigDefaults

Se os defaults não estiverem em conformidade com o schema, a inicialização falha.

Observação:
Acessar e utilizar diretamente métodos e atributos de `ConfigManager` só é necessário caso
seja preciso mais funcionalidade além do que `ConfigInterface` oferece. Em uso típico,
classes de alto nível devem herdar de `ConfigInterface` e evitar interagir com
`ConfigManager` diretamente.

---

## 3. O que `ConfigInterface` carrega por padrão

`ConfigInterface` é a interface de alto nível padrão utilizada por classes de ETL e modelagem.
Ela encapsula o `ConfigManager`, aplica defaults e expõe blocos de configuração frequentemente utilizados como atributos.

Configurações ETL carregadas:

    etl/paths.yml                       -> self.paths_cfg
    etl/files_and_folders.yml           -> self.files_and_folders_cfg
    etl/fields.yml                      -> self.fields_cfg

    etl/followup/<DEFAULT>.yml          -> self.followup_cfg
    etl/birads_classifier/<DEFAULT>.yml -> self.birads_clf_cfg
    etl/bmi_model/<DEFAULT>.yml         -> self.bmi_models_cfg
    etl/embeddings/<DEFAULT>.yml        -> self.embeddings_cfg

Configurações de modelo carregadas:

    model/extra.yml                          -> self.extra_cfg
    model/fields.yml                         -> self.model_fields_cfg
    model/splits/<DEFAULT>.yml               -> self.split_cfg
    model/training_experiments/<DEFAULT>.yml -> self.training_cfg
    model/tuning/<DEFAULT>.yml               -> self.tuning_cfg
    model/eval/<DEFAULT>.yml                 -> self.eval_cfg   (se configurado)

Isso significa que: se o esquema de pastas estiver presente e os defaults forem válidos,
o pipeline completo consegue localizar todas as configurações necessárias por meio dessa interface.

---

## 4. Resolução de caminhos e IDs de execução

O arquivo etl/paths.yml deve definir pelo menos:

- extract.path e extract.id
- transform.path e transform.id
- load.path e load.id
- (opcional) entradas logging_path

`ConfigInterface` combina caminhos base com IDs para criar diretórios específicos por execução:

    extract_path   = extract.path   / extract.id
    transform_path = transform.path / transform.id
    load_path      = load.path      / load.id

Também define caminhos derivados como:

    transformed_data_path = transform_path / "transformed"
    dataset_path = load_path / followup_cfg["dataset_name"]   (se followup estiver configurado)

Esses IDs são como o pipeline separa diferentes execuções/coleções no disco.

---

## 5. Nomes de pastas esperados e efeitos colaterais (importante)

`ConfigInterface` lê nomes de pastas de etl/files_and_folders.yml e cria
múltiplos diretórios na inicialização.

Isso possui efeitos colaterais:

- Diretórios são criados automaticamente se estiverem ausentes.
- Caminhos ou IDs incorretos na configuração criarão diretórios no local errado.
- Sempre verifique paths.yml antes de executar.

Exemplos de diretórios criados:

- Pastas de extração bruta (mammograms, biopsy, anamnesis, patient/person/user)
- Pastas de transformação (saídas de birads, modelos ajustados, cache, armazenamento de embeddings)
- Pasta de load e pasta de dataset

---

## 6. Checklist mínimo para um novo diretório de configuração

Para criar uma nova pasta de configuração (nova coleção), certifique-se de que:

1) A estrutura de pastas exista:

    config_dir/<br>
      etl/<br>
      model/<br>

2) Arquivos base obrigatórios existam:

    etl/paths.yml<br>
    etl/files_and_folders.yml<br>
    etl/fields.yml<br>
    model/extra.yml<br>
    model/fields.yml<br>

3) Subpastas obrigatórias existam (mesmo que inicialmente com apenas um config):

    etl/followup/<br>
    etl/embeddings/<br>
    etl/birads_classifier/<br>
    etl/bmi_model/<br>
    model/splits/<br>
    model/training_experiments/<br>
    model/tuning/<br>

4) O mapeamento de defaults selecione nomes de arquivos válidos que existam nessas pastas.

5) Todos os caminhos em paths.yml apontem para locais válidos com permissão de escrita.

---

## 7. Padrão prático de uso

Uso recomendado:

- Criar um novo diretório de configuração para cada “coleção de execução” (por exemplo, por data ou lote de estudo).
- Manter todas as variantes experimentais como arquivos YAML separados em model/training_experiments/ e model/tuning/.
- Versionar a pasta de configuração (ou pelo menos os YAMLs) para garantir que os experimentos sejam reprodutíveis.

A pasta de configuração faz parte do registro científico de cada conjunto de experimentos.

---

## 8. Armadilhas comuns e proteções

1) Colisão de stems  
   Se você criar dois arquivos com o mesmo stem na mesma pasta, um sobrescreverá o outro em memória.
   Sempre mantenha stems únicos por pasta.

2) Defaults ausentes  
   Se `config_defaults` referenciar um nome de arquivo que não exista, o default se tornará um stem que não poderá
   ser resolvido posteriormente. Certifique-se de que os defaults sempre apontem para arquivos reais no diretório de configuração.

3) Caminhos `None` silenciosos no acesso  
   A função auxiliar `access_nested_dict` retorna `None` imediatamente se qualquer chave no caminho for `None`.
   Se permitir defaults opcionais (por exemplo, eval = null), assegure que o código downstream trate configs ausentes.

4) Diretórios com efeito colateral  
   `ConfigInterface` cria diretórios na inicialização.
   Sempre valide que etl/paths.yml aponta para a raiz de saída correta antes de executar.

5) Reprodutibilidade: sempre salve snapshot das configurações  
   Para cada execução, copie o diretório completo de configuração (ou pelo menos os arquivos YAML utilizados) para a pasta de saída.
   Isso evita que “config drift” comprometa a reprodutibilidade.

---

## 9. Melhoria sugerida (opcional): snapshot de metadados de execução

Um padrão recomendado para rastreamento de experimentos:

- Criar uma pasta de saída para cada execução (extract_id, transform_id, load_id).
- Salvar uma cópia de:
    - o mapeamento de defaults utilizado
    - os stems resolvidos
    - os YAMLs relevantes (ou a pasta completa de configuração)
    - o hash do commit git

Isso pode ser implementado como uma pequena função auxiliar chamada no início de cada execução via CLI.

Isso não é estritamente necessário, mas torna os resultados auditáveis e reproduzíveis.

---

# Schemas para os arquivos de configuração

Isso pode ser melhorado, se julgado necessário.

## etl/paths.yml

    extract:
        id: str
        path: str
        env_path: str # path to .env
        logging_path: str
    transform:
      id: str
      path: str
      logging_path: str
    load:
      id: str
      path: str
      logging_path: str

## etl/files_and_folders.yml

    extract:
      folders:
        biopsy: str
        mammogram_exams: str
        anamnesis: str
        user_person_data: str
    transform:
      folders:
        fitted_models: str
        embedding: str
        embedding_store: str
        birads: str
        birads_model: str
        cache: str
      transformed_files:
        anamnesis: str
        person: str
        user: str
        similarity_data: str
        valid_person_patient: str
        breast_biopsy: str
        breast_biopsy_classified: str 
        person_biopsy: str
        person_mammogram: str
    load:
      load_files:
        merged_data: str # with .parquet extension
        seq_per_mammogram_filename: str # with .parquet extension
        final_data_before_eligibility_filename: str # with .parquet extension
        final_data_with_eligibility_filename: str # with .parquet extension
        precomputed_filename: str # with .lmdb extension

## etl/fields.yml (legacy)

    fields:
      person_id: CD_PESSOA
      patient_id: CD_PACIENTE
      user_id: CD_USUARIO
      mammogram_id: CD_ATENDIMENTO
      mammogram_id_final: key
      person_birthdate: DT_NASCIMENTO_FUNDACAO
      mammogram_date: DT_ATENDIMENTO
      mammogram_text: DS_LAUDO_MEDICO
      anamnesis_id: CD_ATENDIMENTO
      anamnesis_date: DT_ATENDIMENTO
      anamnesis_raw_features: [ 
        DS_MENARCA, DS_MENOPAUSA, NU_GESTACAO_, NU_GESTACAO_ABORTO, FL_CA_MAMA_MAE, 
        FL_CA_MAMA_IRMA, FL_CA_MAMA_AVO, FL_CA_MAMA_TIA, FL_MASTECTOMIA_MD, 
        FL_MASTECTOMIA_ME, FL_PLASTICA_ME , DT_PLASTICA_ME, FL_PLASTICA_MD , 
        DT_PLASTICA_MD, DT_ATENDIMENTO, CD_ATENDIMENTO, FL_ALEITAMENTO 
      ] 

## etl/birads_classifier/*.yml

    phase: transform
    birads_classifier:
      max_samples_per_class: int
      val_size: float
      split_random_state: int
      tfidf_max_features: int
      ngram_range_max: in
      clf_penalty: str
      clf_solver: str
      re_processed_filename: str # with .parquet extension
      ml_infered_filename: str # with .parquet extension

## etl/followup/*.yml

    phase: load
    dataset_name: str
    followup:
      minimum_age: int
      maximum_age: int
      grace_period_start_in_days: int # -- set an interval of D days after the index mammogram so that we do not consider follow-up in this period.
      total_months_of_followup: 120 # keep
      start_date_mammogram: str # YYYY-MM-DD format
      cohort_end_date: str # YYYY-MM-DD format
      birads_5:
        validation_version: 1
        validation_interval_months: 6
        validation_explanation: [
          v1 -> keeps BI-RADS 5 only if within an interval of time after its date there isn't any benign BI-RADS or benign biopsy.,
          v2 -> keeps BI-RADS 5 if there is either a BI-RADS 6 or confirmatory biopsy after its date.
        ]
    precomputed:
      path: str
      filename: str (with extension)

## etl/embeddings/*.yml

    phase: transform
    embedding_id: str
    tfidf: # applies only when using tf-idf
      svd: bool
      svd_dim: int
      max_features: int
      min_df: int
      max_df: float
      ngram_range: list[int]
      model_name: str

## etl/bmi_model/*.yml (to be removed)

    bmi_model:
        path: str
        linreg_model: str # with .pkl extension
        randfor_model: str # with .pkl extension

## model/fields.yml

    fields:
        feature_columns: [
          mammogram_id, mammogram_current_result, monthly_payment_min, monthly_payment_max, 
          bmi, menarche_age, age_at_first_mammogram, age_at_mammogram, ca_mama_mae_cat_-1.0, 
          ca_mama_mae_cat_0.0, ca_mama_mae_cat_1.0, ca_mama_irma_cat_-1.0, ca_mama_irma_cat_0.0, 
          ca_mama_irma_cat_1.0, ca_mama_avo_cat_-1.0, ca_mama_avo_cat_0.0, ca_mama_avo_cat_1.0, 
          ca_mama_tia_cat_-1.0, ca_mama_tia_cat_0.0, ca_mama_tia_cat_1.0, menopause_category_ordered, 
          is_missing_children, is_missing_miscarriage, number_of_children, number_of_miscarriage, 
          zipcode_embedding_0, zipcode_embedding_1, zipcode_embedding_2, zipcode_embedding_3, 
          zipcode_embedding_4, zipcode_embedding_5, zipcode_embedding_6, zipcode_embedding_7, 
          breastfeeding_cat
        ] # fixed for now
        event_indicator_columns: [
          event_indicator_1yr, event_indicator_2yr, event_indicator_3yr, 
          event_indicator_4yr, event_indicator_5yr, event_indicator_6yr,
          event_indicator_7yr, event_indicator_8yr, event_indicator_9yr,
          event_indicator_10yr
        ] # fixed
        followup_columns: [
          14days_1yr_followup, 1yr_2yr_followup, 2yr_3yr_followup, 
          3yr_4yr_followup, 4yr_5yr_followup, 5yr_6yr_followup,
          6yr_7yr_followup, 7yr_8yr_followup, 8yr_9yr_followup,
          9yr_10yr_followup
        ] # fixed
        multiyear_eligibility_columns: [
          eligibility_0yr_1yr, eligibility_1yr_2yr, eligibility_2yr_3yr, 
          eligibility_3yr_4yr, eligibility_4yr_5yr, eligibility_5yr_6yr,
          eligibility_6yr_7yr, eligibility_7yr_8yr, eligibility_8yr_9yr,
          eligibility_9yr_10yr
        ] # fixed
        birads_column: str

## model/extra.yml

    extra:
        checkpoint_path: str
        logging_path: str
        tuning_path: str
        eval_path: str
        use_amp: true
        device: str # either 'cpu' or 'cuda'
        verbose: bool
        save_epochs: bool
        save_best_epochs: bool

## model/split/*.yml

    description: str
    split:
      training_size: float # < 1.0
      test_size: float # sums to 1.0 with training_size
      kfold: int
      seed: int
      birads: list[int] # e g. [1,2,3] 

## model/tuning/*.yml

    description: tuning for several age strats and different target years. 
    model:
      mammogram_input_dim: 5000
      extra_features_dim: 33
      embed_dim: 128
      transformer_num_heads: int # not needed for tf-idf embeddings
      transformer_num_layers: int # not needed for tf-idf embeddings
      transformer_dropout: float # not needed for tf-idf embeddings
      freeze_transformer: bool
      sigmoid_output: bool
      mlp_config:
        hidden_layers: list[int]
        dropout: float
        activation: str  # options: relu, gelu and mish
        use_batchnorm: true
        sigmoid: false

    training:
      epochs: int
      early_stop: bool
      patience: int
      learning_rate: float
      weight_decay: float
      warmup_steps: int
      loss_function: cross_entropy
      optimizer: str # 'adam' or 'sgd'
      max_training_batches_per_epoch: int
      max_validation_batches_per_epoch: int

    tuning:
      path: str # for logging
      num_trials: int
      optim_seed: int
      study_name: str


## model/training_experiments/*.yml

    model:
      mammogram_input_dim: 5000
      extra_features_dim: 33
      embed_dim: 128
      transformer_num_heads: int # not needed for tf-idf embeddings
      transformer_num_layers: int # not needed for tf-idf embeddings
      transformer_dropout: float # not needed for tf-idf embeddings
      freeze_transformer: bool
      sigmoid_output: bool
      mlp_config:
        hidden_layers: list[int]
        dropout: float
        activation: str  # options: relu, gelu and mish
        use_batchnorm: true
        sigmoid: false
    training:
      epochs: int
      early_stop: bool
      patience: int
      learning_rate: float
      weight_decay: float
      warmup_steps: int
      loss_function: cross_entropy
      optimizer: str # 'adam' or 'sgd'
      max_training_batches_per_epoch: int
      max_validation_batches_per_epoch: int
      model_name: str
      target_year: int
      sampling_strategy: str # either 'undersampling' or 'oversampling'
      negative_to_positive_ratio: int
      num_workers: int
      batch_size: int
      pretrained:
        load: bool
        model_name: str
        file_name: best_model.pt
    description: str

