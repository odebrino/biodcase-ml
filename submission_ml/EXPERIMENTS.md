# Experiments

Referencia atual: `outputs/runs/20260421-223457`.

| Experimento | Config | Data | Macro-F1 | F1 bpd | Macro-F1 casey2017 | Observacao |
| --- | --- | --- | ---: | ---: | ---: | --- |
| baseline limpo | `configs/nitro4060.yaml` | 2026-04-21 | 0.8593 | 0.5193 | 0.7323 | Bom resultado geral, mas `bpd` cai muito para `bmd`. |
| focal bpd | `configs/nitro4060_bpd.yaml` | 2026-04-21 | 0.8866 | 0.7725 | 0.7642 | Melhor run ate agora; melhora `bpd` sem perder Macro-F1. |
| sampler | `configs/nitro4060_sampler.yaml` | nao rodado | - | - | - | Proximo teste se quisermos comparar com FocalLoss. |
| pretrained | `configs/nitro4060_pretrained.yaml` | nao rodado | - | - | - | Pode ajudar se o ganho em `casey2017` travar. |
| global norm | `configs/nitro4060_global_norm.yaml` | nao rodado | - | - | - | Teste separado para ver se reduz diferenca entre datasets. |

## Leitura Rapida

O FocalLoss resolveu boa parte do problema que mais incomodava: `bpd` saiu de F1 `0.5193` para `0.7725`. O run ainda erra `bpd/bmd`, mas agora o maior bloco de confusao aparece em `bmb -> bmz`. `casey2017` tambem subiu, embora continue pior que os dois datasets de Kerguelen.

## Criterios

Um experimento so entra como melhoria se:

- `Macro-F1 >= 0.8593`, ou queda maxima de `0.01` com ganho forte em `bpd`;
- `F1 bpd > 0.60`;
- `Macro-F1 casey2017 > 0.75`;
- o cache nao crescer sem controle.

## Comandos

```bash
python -m src.training.train --config configs/nitro4060_bpd.yaml --manifest data_manifest.csv
python -m src.training.evaluate --checkpoint outputs/runs/<run>/best_model.pt --config configs/nitro4060_bpd.yaml --manifest data_manifest.csv --output-dir outputs/runs/<run>
```

Para olhar o cache:

```bash
python -m src.data.cache_tools --summary
```

Para exportar exemplos dos erros `bpd`/`bmd`:

```bash
python -m src.analysis.inspect_errors \
  --report outputs/runs/<run>/bpd_error_report.csv \
  --config configs/nitro4060.yaml \
  --out outputs/error_samples/<run>
```
