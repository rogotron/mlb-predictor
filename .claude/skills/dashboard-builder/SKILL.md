---
name: dashboard-builder
description: >-
  Governs edits to the Diamond Forecast React/Vite dashboard (root src/App.tsx
  and src/components/*, which some notes call "DiamondEdgeAI"). Use this skill
  whenever changing, adding, reviewing, or debugging dashboard UI: matchup
  cards, the game-detail panel, the Factors view, the provenance/freshness
  header, win-probability displays, or anything that shows model predictions or
  data freshness. It enforces the core invariant that every displayed value
  traces to the slate artifact, with no mock, hardcoded, or client-recomputed
  numbers.
---

# Dashboard builder

The Diamond Forecast dashboard exists to show what the trained model actually
produced. Its credibility depends on one rule: what the user sees on screen is
what the pipeline wrote to disk. This skill keeps edits from quietly breaking
that contract, because the ways to break it (mock data, `Date.now()`,
recomputing numbers in JS) all look harmless in a diff.

## The data contract

The React app reads two things: the live MLB schedule API (for the game list)
and the static slate artifact `public/slates/{date}.json` (for everything the
model produced). It does not call the FastAPI backend. Read
`src/services/modelSlate.ts` before editing; it is the contract.

Flow, with the real symbols:

- `fetchModelSlate(date)` → `ModelSlateResult { status, predictionsByGamePk,
  provenance, model, generatedAt }`. This is the only door for model data.
- Per game, `buildPredictionFromModelSlate(game, staticGame)` builds the
  `GamePrediction`. Its `topFactors` come from `factorsFromSlate(staticGame)`,
  which maps `staticGame.factors[]` (real LightGBM importance shares). 
- The slate-level `model.factorGroups` feeds the Factors tab and the rail via
  `modelSummaryToFactors(modelSummary)` in `src/App.tsx` (`modelFactorList`,
  gated by `factorsAreReal`).
- `provenance` drives `src/components/ProvenanceBar.tsx` and the header
  "Updated" chip through `formatAsOfLabel(provenance.dataAsOf)`.

The slate JSON's shape (top level): `date`, `count`, `generatedAt`,
`provenance` (per-source freshness), `model` (importances by group), `games[]`.
Each game has `game`, `away`, `home`, `prediction`, `pitchers`, `stats`,
`factors[]`, `runDist`, and `explain: {shap: null}` (reserved extension point).

## Hard rules

These are the invariants. Each one has a specific failure it prevents.

1. **Freshness comes from the manifest, never the clock.** The "Updated" label
   is `formatAsOfLabel(provenance.dataAsOf)` (the build time). Do not use
   `new Date()` / `Date.now()` for it. `lastUpdatedLabel()` (browser now) is a
   fallback only for when no slate is loaded. A clock-based timestamp implies
   freshness that the data may not have.

2. **Factors render from the slate, not from JS.** Per-game drivers come from
   `staticGame.factors[]`; the Factors tab and rail come from
   `model.factorGroups`. Never resurrect the old `buildDifferentiatingFactors`
   pattern (scaling `staticGame.stats` with hardcoded constants), and never feed
   the mock `modelFactors` into a view that has a real slate loaded. The slate's
   factor percentages are real model gain shares; anything recomputed client
   side is a fabrication that will drift from the model.

3. **Never reintroduce synthetic factors.** The producer
   (`_build_factors` in `backend/services/assemble.py`, grouped by
   `src/models/feature_groups.py`) only emits groups the deployed model actually
   uses. The removed offenders were a fabricated "Home Field Factor"
   (`win_pct x 0.80`) and an anchored "Lineup Matchup" (`win_pct x 0.50`). Do
   not add these back on either side. If a group has zero model gain it must not
   appear.

4. **No weather, no umpire in the UI.** `GamePrediction` has no `weather` field.
   There is no "Weather impact" edge filter and no "Weather and park" card. The
   `pregame_safe` model has no weather or umpire features, so they appear only
   as `not_collected` chips in `ProvenanceBar`. Do not add a weather field,
   filter, card, or placeholder string.

5. **A displayed number must have a slate source.** If you want to show a value,
   it has to exist in `public/slates/{date}.json`. If it does not, add it to the
   Python assembler first (`_assemble_game` in `backend/services/assemble.py`),
   rebuild the slate, then read it in `modelSlate.ts`. Do not compute it in the
   component. The mock path (`generateMockPredictionForGame`, `predictionSource:
   'prototype'`, and the `modelFactors` list in `src/data/mockModelData.ts`) is
   only for dates with no slate file, and is labeled as prototype in the UI.

## Where things live

- `src/App.tsx` — page shell and state wiring (`modelSummary`,
  `slateProvenance`, `factorsAreReal`, `modelFactorList`, `updatedLabel`), tabs,
  edge filters.
- `src/services/modelSlate.ts` — slate fetch, types, and the factor mapping.
  The data contract; change it deliberately.
- `src/components/ProvenanceBar.tsx` — freshness header (source rows, date
  range, stale/missing badges).
- `src/components/MatchupCard.tsx`, `GameDetailPanel.tsx`,
  `FactorImpactList.tsx` — render prediction and factors.
- `src/data/mockModelData.ts` — fallback only.
- Producer side (edit here when the slate lacks a field): 
  `scripts/build_static_slate.py` → `backend/services/assemble.py`
  (`_assemble_game`, `_build_factors`, `build_model_factor_summary`),
  `src/data/provenance.py`, `src/models/feature_groups.py`.

## Before you finish an edit

Run through this. It is fast and it catches the failures the rules describe.

- `npx tsc --noEmit` is clean.
- Every new or changed on-screen value traces to a field in the slate JSON.
  If you cannot point to the field, it is a bug.
- The "Updated" label still resolves from `provenance.dataAsOf`.
- Grep the changed components for `Date.now(`, `new Date(`, `Math.random`, and
  literal number arrays. Each hit in a model-data path is suspect.
- Start the preview, load a date that has a slate, confirm `/slates/{date}.json`
  returns 200, and check the console for errors. Verify the provenance bar,
  the factors, and one matchup card show real values, not the prototype
  fallback.

## When a new feature reaches the model

If someone adds a model feature and wants it on the dashboard, the honest path
is: classify it in `diagnostics/feature_inventory.csv`, map it to a group in
`src/models/feature_groups.py`, and it will flow into the slate's `factors[]`
and `model.factorGroups` automatically. The dashboard then shows it with no
hardcoding. That is the whole point of the contract: the UI never invents, it
reflects.
