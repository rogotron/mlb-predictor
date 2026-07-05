import type { GamePrediction, ModelFactor, PitcherDetail } from '../data/mockModelData';
import type { MlbGame } from './mlbScheduleApi';

type StaticSlateTeam = {
  full?: string;
  abbr?: string;
  record?: string;
};

type StaticSlatePrediction = {
  winner?: string;
  awayProb?: number;
  homeProb?: number;
  awayRuns?: number;
  homeRuns?: number;
  confidence?: number;
  confLabel?: string;
  spread?: string;
  total?: string;
  source?: string;
  drivers?: string[];
};

export type SlateFactor = {
  name?: string;
  pct?: number;
  note?: string;
  source?: string;
};

type StaticSlateGame = {
  game?: {
    gamePk?: number;
    venue?: string;
    firstPitch?: string;
    status?: string;
  };
  away?: StaticSlateTeam;
  home?: StaticSlateTeam;
  prediction?: StaticSlatePrediction;
  pitchers?: {
    away?: PitcherDetail;
    home?: PitcherDetail;
  };
  stats?: Array<{
    stat?: string;
    away?: string;
    home?: string;
  }>;
  factors?: SlateFactor[];
  // Extension point for per-game SHAP attributions (not yet populated).
  explain?: { shap?: unknown };
};

export type SlateSource = {
  key: string;
  label: string;
  status: 'ok' | 'missing' | 'error' | 'not_collected';
  rows: number;
  minDate: string | null;
  maxDate: string | null;
  daysBehind: number | null;
  stale: boolean;
};

export type SlateProvenance = {
  dataAsOf?: string;
  targetDate?: string;
  gameCount?: number | null;
  dateRange?: { start?: string | null; end?: string | null };
  anyStale?: boolean;
  anyMissing?: boolean;
  sources?: SlateSource[];
};

export type SlateModelGroup = { name: string; source: string; pct: number };

export type SlateModelSummary = {
  name?: string;
  mode?: string;
  featureCount?: number;
  importanceMetric?: string;
  factorGroups?: SlateModelGroup[];
};

type StaticSlatePayload = {
  date?: string;
  generatedAt?: string;
  provenance?: SlateProvenance;
  model?: SlateModelSummary;
  games?: StaticSlateGame[];
};

export type ModelSlateStatus = 'loaded' | 'missing' | 'error';

export type ModelSlateResult = {
  status: ModelSlateStatus;
  predictionsByGamePk: Map<number, StaticSlateGame>;
  provenance?: SlateProvenance;
  model?: SlateModelSummary;
  generatedAt?: string;
  message?: string;
};

function normalizeConfidence(label?: string): GamePrediction['confidenceLabel'] {
  const value = (label ?? '').toLowerCase();
  if (value.includes('high')) return 'High';
  if (value.includes('medium') || value.includes('moderate')) return 'Medium';
  return 'Low';
}

function factorFromDriver(driver: string, index: number): ModelFactor {
  const lower = driver.toLowerCase();
  const category = lower.includes('bullpen')
    ? 'Bullpen'
    : lower.includes('starter') || lower.includes('pitcher')
      ? 'Pitching'
      : lower.includes('run') || lower.includes('record')
        ? 'Form'
        : 'Model';

  return {
    name: index === 0 ? 'Model pick' : category === 'Pitching' ? 'Starting pitcher edge' : category === 'Bullpen' ? 'Bullpen strength' : 'Team context',
    description: driver,
    impact: Math.max(42, 84 - index * 12),
    direction: index === 0 ? 'Positive' : 'Neutral',
    category,
    exampleSignal: driver,
    affects: index === 0 ? ['Win probability', 'Confidence'] : ['Win probability'],
  };
}

function factorsFromDrivers(drivers: string[]): ModelFactor[] {
  const usedNames = new Set<string>();
  return drivers.slice(0, 3).map((driver, index) => {
    const factor = factorFromDriver(driver, index);
    let name = factor.name;
    for (let suffix = 2; usedNames.has(name); suffix += 1) {
      name = `${factor.name} (${suffix})`;
    }
    usedNames.add(name);
    return { ...factor, name };
  });
}

function categoryForFactor(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('bullpen')) return 'Bullpen';
  if (lower.includes('pitcher') || lower.includes('pitch')) return 'Pitching';
  if (lower.includes('form')) return 'Form';
  if (lower.includes('run production') || lower.includes('record')) return 'Offense';
  if (lower.includes('rest') || lower.includes('availability')) return 'Schedule';
  if (lower.includes('park')) return 'Context';
  return 'Model';
}

// Maps the slate's model-derived factor groups (real LightGBM importance
// shares) into the dashboard's ModelFactor shape. Importance is a magnitude,
// not a per-game direction, so every factor is Neutral until SHAP adds signed
// per-game attributions via the game `explain` extension point.
export function factorFromSlate(slateFactor: SlateFactor): ModelFactor {
  const name = slateFactor.name ?? 'Model factor';
  const pct = Number(slateFactor.pct ?? 0);
  const source = slateFactor.source ?? 'model';
  return {
    name,
    description: slateFactor.note ? `${slateFactor.note} — ${source}.` : `${source}.`,
    impact: Math.round(pct),
    direction: 'Neutral',
    category: categoryForFactor(name),
    exampleSignal: slateFactor.note ?? source,
    affects: ['Win probability', 'Confidence'],
  };
}

function factorsFromSlate(staticGame: StaticSlateGame): ModelFactor[] {
  const factors = (staticGame.factors ?? [])
    .filter((f) => typeof f.pct === 'number')
    .map(factorFromSlate);
  if (factors.length) return factors;
  return factorsFromDrivers(staticGame.prediction?.drivers ?? []);
}

export function modelSummaryToFactors(model?: SlateModelSummary): ModelFactor[] {
  return (model?.factorGroups ?? []).map((group) =>
    factorFromSlate({ name: group.name, pct: group.pct, source: group.source }),
  );
}

function buildModelExplanation(
  game: MlbGame,
  staticGame: StaticSlateGame,
  confidenceLabel: GamePrediction['confidenceLabel'],
  topFactors: ModelFactor[],
) {
  const prediction = staticGame.prediction;
  const winner = prediction?.winner ?? (Number(prediction?.homeProb ?? 50) >= 50 ? game.homeTeam.name : game.awayTeam.name);
  const winProb = Math.max(Number(prediction?.awayProb ?? 50), Number(prediction?.homeProb ?? 50));
  const mainFactor = topFactors[0];
  const supportFactor = topFactors[1];
  const risk =
    confidenceLabel === 'High'
      ? 'The main offsetting risk is baseball variance in late innings, though the model still rates this as a high-confidence spot.'
      : 'The main offsetting risk is that the edge is modest, which keeps the model confidence from moving higher.';

  return [
    `The trained model gives ${winner} a ${winProb}% win probability for this matchup.`,
    mainFactor
      ? `The biggest differentiating factor is ${mainFactor.name.toLowerCase()}: ${mainFactor.description}`
      : 'The biggest differentiating factor is the combined model feature stack.',
    supportFactor
      ? `A secondary separator is ${supportFactor.name.toLowerCase()}: ${supportFactor.description}`
      : `The projected run environment is ${prediction?.total ?? 'near league average'} total runs based on the current model slate.`,
    risk,
  ].join(' ');
}

export async function fetchModelSlate(date: string): Promise<ModelSlateResult> {
  try {
    const response = await fetch(`/slates/${date}.json`, { cache: 'no-store' });
    if (response.status === 404) {
      return {
        status: 'missing',
        predictionsByGamePk: new Map(),
        message: `No trained-model static slate exists for ${date}. Run python scripts/build_static_slate.py --date ${date}.`,
      };
    }
    if (!response.ok) {
      return {
        status: 'error',
        predictionsByGamePk: new Map(),
        message: `Model slate request failed with HTTP ${response.status}.`,
      };
    }

    const payload = (await response.json()) as StaticSlatePayload;
    const predictionsByGamePk = new Map<number, StaticSlateGame>();
    (payload.games ?? []).forEach((game) => {
      const gamePk = game.game?.gamePk;
      if (typeof gamePk === 'number') {
        predictionsByGamePk.set(gamePk, game);
      }
    });

    return {
      status: 'loaded',
      predictionsByGamePk,
      provenance: payload.provenance,
      model: payload.model,
      generatedAt: payload.generatedAt,
    };
  } catch (error) {
    return {
      status: 'error',
      predictionsByGamePk: new Map(),
      message: error instanceof Error ? error.message : 'Unable to load trained-model static slate.',
    };
  }
}

export function buildPredictionFromModelSlate(game: MlbGame, staticGame: StaticSlateGame): GamePrediction | null {
  const prediction = staticGame.prediction;
  if (!prediction) return null;

  const awayWinProbability = Number(prediction.awayProb ?? 50);
  const homeWinProbability = Number(prediction.homeProb ?? 50);
  const confidenceLabel = normalizeConfidence(prediction.confLabel);
  const drivers = prediction.drivers ?? [];
  const confidence = Number(prediction.confidence ?? Math.abs(homeWinProbability - 50) * 2);
  const topFactors = factorsFromSlate(staticGame);

  return {
    id: game.id,
    gamePk: game.gamePk,
    date: game.date,
    time: game.timeET,
    venue: staticGame.game?.venue ?? game.venue,
    status: game.status,
    sourceGame: game,
    awayTeam: {
      ...game.awayTeam,
      record: staticGame.away?.record ?? game.awayTeam.record,
    },
    homeTeam: {
      ...game.homeTeam,
      record: staticGame.home?.record ?? game.homeTeam.record,
    },
    awayPitcher: staticGame.pitchers?.away?.name ?? game.awayPitcher ?? 'TBD',
    homePitcher: staticGame.pitchers?.home?.name ?? game.homePitcher ?? 'TBD',
    awayWinProbability,
    homeWinProbability,
    projectedAwayRuns: Number(prediction.awayRuns ?? 0),
    projectedHomeRuns: Number(prediction.homeRuns ?? 0),
    confidenceScore: Math.round(Math.min(100, Math.max(0, confidence * 10))),
    confidenceLabel,
    modelEdge: Number((Math.max(awayWinProbability, homeWinProbability) - 50).toFixed(1)),
    marketLine: prediction.spread ? `Model spread ${prediction.spread}` : 'Trained model slate',
    topFactors: topFactors.length ? topFactors : factorsFromDrivers(drivers),
    riskFactors: [
      'Lineups and late bullpen availability may still change before first pitch.',
      'Scratches and market movement can narrow the edge.',
    ],
    explanation: buildModelExplanation(game, staticGame, confidenceLabel, topFactors),
    predictionSource: 'model',
    pitcherDetails: staticGame.pitchers,
  };
}
