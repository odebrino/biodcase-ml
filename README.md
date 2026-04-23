# BIODCASE Bioacoustic Classification

Pipeline para classificar eventos bioacusticos anotados no
`biodcase_development_set`.

O caminho cientifico principal agora e classico / nao-convolucional:
representacoes explicitas de crops tempo-frequencia alimentam regressao
logistica, SVMs, KNN, Naive Bayes, arvores/boosting e MLP. O caminho CNN antigo
permanece no repositorio apenas como comparacao historica.

## Caminho Principal

```bash
python -m src.classical.baselines \
  --config configs/classical_baselines.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/classical
```

Esse driver compara modelos por familia de representacao, ajusta
normalizacao/PCA somente no subconjunto de treino interno e usa os dominios
oficiais held-out apenas para avaliacao final.

## Resultado CNN Historico

Melhor run CNN historico: `outputs/runs/20260421-223457`.

| Metrica | Valor |
| --- | ---: |
| Accuracy | `0.9301` |
| Macro-F1 | `0.8866` |
| Weighted-F1 | `0.9309` |
| F1 `bpd` | `0.7725` |
| Macro-F1 `casey2017` | `0.7642` |

Esse run usa `configs/nitro4060_bpd.yaml`, com FocalLoss e peso maior para `bpd`
e `bmz`. Ele nao e mais o caminho metodologico principal, pois o brief exige
comparacao classica / nao-convolucional.

## Classes

`label` e o identificador canonico interno. `label_raw` preserva o valor
original da coluna `annotation` para rastreabilidade.

| Classe canonica | Aliases normalizados | Label interno |
| --- | --- | --- |
| Bm-A | `bma`, `BmA`, `Bm-A` | `bma` |
| Bm-B | `bmb`, `BmB`, `Bm-B` | `bmb` |
| Bm-Z | `bmz`, `BmZ`, `Bm-Z` | `bmz` |
| Bm-D | `bmd`, `BmD`, `Bm-D` | `bmd` |
| Bp-20 | `bp20`, `Bp20`, `Bp-20` | `bp20` |
| Bp-20Plus | `bp20plus`, `Bp20plus`, `Bp-20Plus` | `bp20plus` |
| Bp-40Down | `bpd`, `BpD`, `Bp-40Down` | `bpd` |

## Semantica Dos Splits

O diretorio ou valor de manifesto chamado `validation` neste repositorio e um
alias operacional legado para os dominios oficiais de teste: `casey2017`,
`kerguelen2014` e `kerguelen2015`. Ele nao deve ser descrito como uma
validacao generica. Qualquer validacao interna para escolha de modelo deve ser
criada separadamente a partir de `train`; `selection_split: null` deixa claro
que essa divisao interna ainda nao esta materializada no manifesto atual.

## Baseline Antigo

Os artefatos originais na raiz reportam:

- Accuracy: `0.9135`
- Macro-F1: `0.7629`
- Weighted-F1: `0.9113`
- Melhor epoca: `9`

Esses numeros vieram de um pipeline externo baseado em imagens processadas fora desta pasta, entao devem ser comparados com cuidado.

## Nitro V15 / RTX 4060

Setup recomendado:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-cu124.txt
pip install -r requirements-dev.txt
```

`requirements.txt` contem as dependencias Python comuns, `requirements-cu124.txt` instala
`torch`, `torchvision` e `torchaudio` para CUDA 12.4, e `requirements-dev.txt`
adiciona ferramentas de teste como `pytest`.

Verifique a GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Se o ambiente nao tiver CUDA, os scripts caem para CPU e avisam no terminal.

## Manifesto Limpo

```bash
python -m src.data.build_manifest \
  --data-root biodcase_development_set \
  --out data_manifest.csv \
  --quality-report outputs/data_quality_report.csv \
  --quality-summary outputs/data_quality_summary.csv \
  --min-valid-seconds 0.5
```

O manifesto salva duracao real do WAV, tempos clipados, label normalizado,
`label_raw` e status de qualidade. Eventos fora do audio, duplicatas e trechos
com pouco sinal real sao descartados e registrados no relatorio.

A coluna `annotation` dos CSVs de origem e a classe do evento. As colunas
`start_datetime`, `end_datetime`, `low_frequency` e `high_frequency` definem as
coordenadas anotadas do evento no plano tempo-frequencia.

## Treino CNN Historico

Config historica para o Nitro:

```bash
python -m src.training.train \
  --config configs/nitro4060.yaml \
  --manifest data_manifest.csv
```

Defaults importantes:

- `device: cuda`
- `batch_size: 64`
- `num_workers: 4`
- `mixed_precision: true`
- `cache.enabled: true`
- cache em `processed_cache/`

Cada run fica em `outputs/runs/<timestamp>/` com checkpoints, historico,
metricas, matriz de confusao, predicoes e `run_metadata.json`.

Para reproduzir o melhor run CNN historico:

```bash
python -m src.training.train \
  --config configs/nitro4060_bpd.yaml \
  --manifest data_manifest.csv
```

## Avaliacao

```bash
python -m src.training.evaluate \
  --checkpoint outputs/runs/<timestamp>/best_model.pt \
  --config configs/nitro4060.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/evaluation
```

A avaliacao gera:

- `best_metrics.json`
- `classification_report.csv`
- `metrics_by_dataset.csv` com `macro_f1` e `macro_f1_present_classes`
- `metrics_by_dataset_class.csv`
- `baseline_metrics.csv`
- `class_confidence_analysis.csv`
- `pr_curves.csv`
- `error_analysis.csv`
- `top_confusion_pairs.csv`
- `bpd_error_report.csv`
- `bmb_bmz_error_report.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `confusion_matrix.csv`
- `confusion_matrix_normalized.csv`
- `val_predictions.csv`
- `test_predictions.csv` quando o split avaliado e o teste oficial
- `split_metadata.json`

## Inferencia

Por imagem gerada:

```bash
python -m src.training.predict \
  --checkpoint outputs/runs/<timestamp>/best_model.pt \
  --config configs/nitro4060.yaml \
  --image processed_images/validation/bma/example.png
```

Direto de WAV:

```bash
python -m src.training.predict \
  --checkpoint outputs/runs/<timestamp>/best_model.pt \
  --config configs/nitro4060.yaml \
  --audio biodcase_development_set/train/audio/ballenyislands2015/2015-02-04T03-00-00_000.wav \
  --start-seconds 1652.053 \
  --end-seconds 1663.709 \
  --low-frequency 21.9 \
  --high-frequency 28.4
```

## PNGs Para Conferencia

```bash
python -m src.data.make_spectrograms \
  --manifest data_manifest.csv \
  --out processed_images \
  --processed-manifest processed_manifest.csv \
  --max-per-class-per-split 20
```

Esses PNGs nao sao necessarios para treinar.

## Espectrogramas E Recortes

O pipeline agora separa duas semanticas de representacao:

- `time_crop_with_frequency_band_mask`: tensor historico com tres canais
  (espectrograma completo, mascara da banda anotada e banda realcada). Isso e
  uma aproximacao intencional da semantica do enunciado, nao uma equivalencia
  exata.
- `literal time-frequency crop`: recorte do retangulo definido por
  `start_datetime`, `end_datetime`, `low_frequency` e `high_frequency` apos o
  mapeamento para os eixos tempo/frequencia do espectrograma.

As configuracoes atuais continuam disponiveis. Tambem existem presets inspirados
nas recomendacoes APLOSE do brief:

```bash
python -m src.training.train --config configs/aplose_512_98.yaml --manifest data_manifest.csv
python -m src.training.train --config configs/aplose_256_90.yaml --manifest data_manifest.csv
```

Esses presets expõem `nfft`, `winsize` e `overlap` da diretriz. Nas utilidades
classicas novas, eles usam eixos lineares de frequencia para facilitar o crop
literal; o caminho historico de treino conserva o tensor mel com mascara. Eles
nao substituem a pipeline inteira por uma exportacao visual APLOSE literal.

## Representacoes Para ML Classico

As representacoes abaixo geram artefatos `npz` com `X`, `y`, `feature_names` e
metadados por linha. Elas usam o recorte literal tempo-frequencia e nao assumem
CNN.

```bash
python -m src.data.representations \
  --manifest data_manifest.csv \
  --config configs/aplose_512_98.yaml \
  --family patch \
  --img-size 64 \
  --splits train \
  --out outputs/representations/patch_train.npz

python -m src.data.representations \
  --manifest data_manifest.csv \
  --config configs/aplose_512_98.yaml \
  --family handcrafted \
  --splits train \
  --out outputs/representations/handcrafted_train.npz

python -m src.data.representations \
  --manifest data_manifest.csv \
  --config configs/aplose_512_98.yaml \
  --family hybrid \
  --img-size 64 \
  --pca-components 32 \
  --splits train \
  --out outputs/representations/hybrid_train_pca32.npz
```

Familias implementadas:

- `patch`: crop literal, resize, flatten.
- `handcrafted`: duracao, banda anotada, centroid/bandwidth/rolloff espectral,
  energia por sub-banda e contrastes simples.
- `hybrid`: patch flatten-ready concatenado com descritores; com
  `--pca-components`, o PCA reduz a parte de patch antes da concatenacao.

Para conferir se o mapeamento das coordenadas esta plausivel:

```bash
python -m src.data.export_crop_verification \
  --manifest data_manifest.csv \
  --config configs/aplose_512_98.yaml \
  --out outputs/crop_verification \
  --splits train \
  --per-class 2
```

Esse comando salva pequenos paineis comparando o crop literal com a representacao
historica de mascara, alem de um CSV com os limites anotados e os limites dos
bins efetivamente exportados.

## Suite Classica

`configs/classical_baselines.yaml` e o default para o caminho principal. Modelos
suportados:

- `logistic_regression`
- `linear_svm`
- `rbf_svm`
- `knn`
- `gaussian_nb`
- `random_forest`
- `gradient_boosted_trees`
- `mlp`

Artefatos por run em `outputs/classical/<timestamp>/`:

- `official_test_results.csv`
- `official_test_macro_f1_table.csv`
- `official_test_accuracy_table.csv`
- `official_test_best_summary.md`
- `split_strategy.json`
- `grouped_family_mapping.json`
- `all_official_test_predictions.csv`
- `<representation>/<model>/selection_metrics.json`
- `<representation>/<model>/official_test_metrics.json`
- `<representation>/<model>/official_test_grouped_family_metrics.json`
- `<representation>/<model>/official_test_metrics_by_dataset.csv`
- `<representation>/<model>/official_test_grouped_family_metrics_by_dataset.csv`
- `<representation>/<model>/official_test_ambiguity_report.md`
- relatorios CSV por classe, matriz de confusao e predicoes

Mapeamento de familias ambiguas:

- `ABZ`: `bma`, `bmb`, `bmz`
- `DDswp`: `bmd`, `bpd`
- `20Hz20Plus`: `bp20`, `bp20plus`

O split de selecao interna e derivado apenas de `train`. Quando ha grupos de
dataset suficientes, o driver usa `GroupShuffleSplit` por `dataset`; caso
contrario, registra explicitamente o fallback em `split_strategy.json`.

## Testes

```bash
python -m pytest -q
```

## Experimentos

O melhor run atual e `outputs/runs/20260421-223457`.

Problemas principais:

- `bpd` melhorou bastante, mas ainda existem erros `bpd -> bmd` e `bmd -> bpd`;
- `casey2017` melhorou, mas continua abaixo de `kerguelen2014` e `kerguelen2015`;
- `bmb` e `bmz` ainda aparecem muito na analise de erro;
- o cache pode crescer bastante.

Configs prontas para ablacões:

```bash
python -m src.training.train --config configs/nitro4060_bpd.yaml --manifest data_manifest.csv
python -m src.training.train --config configs/nitro4060_sampler.yaml --manifest data_manifest.csv
python -m src.training.train --config configs/nitro4060_pretrained.yaml --manifest data_manifest.csv
python -m src.training.train --config configs/nitro4060_global_norm.yaml --manifest data_manifest.csv
```

Ferramentas uteis:

```bash
python -m src.data.cache_tools --summary
python -m src.data.cache_tools --clear
python -m src.analysis.inspect_errors --report outputs/runs/20260421-223457/bpd_error_report.csv --out outputs/error_samples/20260421-223457
```

Ao comparar datasets que nao contem todas as classes, prefira
`macro_f1_present_classes`; o `macro_f1` classico continua salvo para
compatibilidade, mas inclui F1 zero para classes ausentes naquele dataset.

## Notas De Metodo

Nos primeiros testes, treinar com pesos simples ajudava as classes maiores, mas deixava `bpd` fraco. A troca para FocalLoss com um peso extra moderado em `bpd` aumentou o recall dessa classe sem derrubar o Macro-F1 global. Nao usamos PNGs como entrada principal porque recalcular e salvar todas as imagens deixa os experimentos mais pesados; os tensores em cache representam o mesmo recorte de espectrograma e carregam tambem a mascara da banda anotada. Os PNGs ficam para conferir visualmente se os recortes fazem sentido.

## Notas

- A metrica principal global e Macro-F1; por dataset, use `macro_f1_present_classes`
  quando houver classes ausentes.
- O input padrao tem 3 canais: espectrograma completo, mascara da banda anotada e banda realcada.
- `validation` e mantido como nome de split legado nos arquivos existentes, mas
  representa o teste oficial por dominios site-year.
- Arquivos grandes como WAVs, cache, PNGs e checkpoints ficam fora do Git.
