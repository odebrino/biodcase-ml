# BIODCASE Bioacoustic Classification

Pipeline PyTorch para classificar eventos bioacusticos anotados no `biodcase_development_set`.

O treino principal usa espectrogramas gerados on-the-fly a partir do manifesto limpo. PNGs continuam disponiveis apenas para conferencia visual.

## Resultado Atual

Melhor run ate agora: `outputs/runs/20260421-223457`.

| Metrica | Valor |
| --- | ---: |
| Accuracy | `0.9301` |
| Macro-F1 | `0.8866` |
| Weighted-F1 | `0.9309` |
| F1 `bpd` | `0.7725` |
| Macro-F1 `casey2017` | `0.7642` |

Esse run usa `configs/nitro4060_bpd.yaml`, com FocalLoss e peso maior para `bpd` e `bmz`. A maior melhoria veio em `bpd`, que no baseline era confundido com `bmd`.

## Classes

| Classe do enunciado | Label usado |
| --- | --- |
| BmA | `bma` |
| BmB | `bmb` |
| BmD | `bmd` |
| BmZ | `bmz` |
| Bp20 | `bp20` |
| Bp20plus | `bp20plus` |
| BpD | `bpd` |

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

O manifesto salva duracao real do WAV, tempos clipados e status de qualidade. Eventos fora do audio, duplicatas e trechos com pouco sinal real sao descartados e registrados no relatorio.

## Treino

Config principal para o Nitro:

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

Cada run fica em `outputs/runs/<timestamp>/` com checkpoints, historico, metricas, matriz de confusao, predicoes e `run_metadata.json`.

Para reproduzir o melhor run atual:

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
- Arquivos grandes como WAVs, cache, PNGs e checkpoints ficam fora do Git.
