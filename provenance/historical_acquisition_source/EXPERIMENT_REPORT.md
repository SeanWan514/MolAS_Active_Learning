# Five-Fold Cost-Aware Mean-UCB Experiment Report

## Scope

- 6 conditions: plain Mean-UCB, direct cost, runtime uncertainty, five cost tiers, square-root cost, and logarithmic cost.
- 5 MolAS folds.
- 3 seeds.
- Every integer checkpoint from 1% through 100%.
- 90 completed runs and 9,000 reconstructed scoring matrices.

## Integrity audit

- Input alignment: pass (3,179 complexes, 7 algorithms, 260 combined features).
- Fold sizes: 636/636/636/636/635 held-out complexes.
- Formula tests: pass.
- Resume-idempotence test: pass.
- Complete runs: 90/90.
- Complete matrices: 9,000/9,000.
- Matching cumulative and newly queried logs: 9,000 each.
- Duplicate or count-mismatched query records: 0.
- Hidden score/runtime policy: acquisition receives only predictions trained on already queried cells; truth is held by the oracle and revealed after selection.

## Runtime-feature ablation

Five-fold means:

| Features | Log MAE | Log RMSE | Runtime MAE | Runtime RMSE | Log R2 | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Learned + algorithm | 0.2897 | 0.4509 | 8.2277 | 52.4191 | 0.8114 | 0.9006 |
| Physical + algorithm | 0.1935 | 0.3683 | 7.7737 | 71.6123 | 0.8743 | 0.9417 |
| Combined + algorithm | 0.2588 | 0.4004 | **7.6058** | **51.0630** | 0.8513 | 0.9173 |

Physical descriptors are strongest for log-scale/rank prediction. Combined features give the lowest raw-runtime MAE and RMSE and are used in the main acquisition study, as agreed with the RA.

## Main result at 20% queried cells

Means across 15 fold-seed evaluations:

| Method | Runtime / plain | Unqueried MAE | MAE delta | NDCG@3 delta | Top-1 |
|---|---:|---:|---:|---:|---:|
| Runtime uncertainty | 0.457 | 0.3030 | -0.0029 | -0.0056 | 0.5520 |
| Direct cost | 0.547 | 0.3013 | -0.0045 | +0.0030 | 0.5743 |
| Log cost | 0.594 | 0.3019 | -0.0039 | **+0.0055** | 0.5743 |
| Square-root cost | 0.618 | **0.3010** | **-0.0049** | +0.0047 | **0.5763** |
| Five tiers | 0.677 | 0.3054 | -0.0004 | -0.0057 | 0.5625 |
| Plain Mean-UCB | 1.000 | 0.3058 | 0 | 0 | 0.5249 |

Negative MAE delta is better. Direct, log, and square-root variants reduce runtime by approximately 38-45% while improving all reported golden-matrix quality measures at 20%.

## Budget dependence

- At 10%, runtime uncertainty nearly preserves NDCG while using 64% of plain runtime.
- At 20%, direct, log, and square-root cost provide the clearest joint improvement.
- At 30%, MAE remains better for all cost-aware methods, but NDCG and top-1 trail plain Mean-UCB.
- At 40%, runtime savings remain large, but plain Mean-UCB has better ranking quality.

The evidence therefore supports a low-budget efficiency claim, not blanket superiority at every query percentage.

## Next required step

Send fold-specific 10%, 20%, 30%, and 40% matrices to the RA for black-box MolAS five-fold training/testing. The 20% checkpoint is the primary candidate. Downstream RMSD/PB-valid metrics are required before making the final application-level claim.
