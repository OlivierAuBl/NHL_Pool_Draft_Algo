# Patch 2 — Stage 1 Lookahead V2

Copier le contenu de ce dossier à la racine du repo `nhl_draft_lab` en conservant les chemins.

## Ce que le patch ajoute

- `lookahead2`: optimise le choix actuel + les 2 prochains choix du GM focal.
- Simulation exacte du snake entre les choix focaux, avec adversaires VORP.
- Branching limité au meilleur joueur disponible de chaque catégorie.
- Diagnostic local de chaque choix contre VORP.
- Résultats agrégés par position de draft.

## Test recommandé

PowerShell:

```powershell
nhl-draft backtest `
  --db data/nhl_history.sqlite `
  --focal-strategies vorp lookahead lookahead2 `
  --opponent-strategies vorp `
  --diagnostics `
  --output-dir output/patch2
```

Les valeurs par défaut sont maintenant celles du projet:

- 15 GM
- 10 F / 3 D / 2 G / 1 T

## Fichiers de sortie

- `stage1_backtest_details.csv`
- `stage1_backtest_summary.csv`
- `stage1_backtest_by_slot.csv`

Les nouvelles métriques incluent:

- `divergence_rate`: proportion des choix où la stratégie diffère de VORP.
- `mean_direct_points_delta_when_diverging`: sacrifice/gain immédiat en points réels quand elle diffère.
- `mean_direct_vorp_delta_when_diverging`: sacrifice/gain immédiat en VORP quand elle diffère.

Le score final, le rang moyen et la marge vs field restent les métriques globales qui disent si ces divergences sont réellement payantes.

## Tests

```powershell
python -m pytest
```

Le patch a été validé avec 15 tests passants.
