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

type StaticSlateGame = {
  game?: {
    gamePk?: number;
    venue?: string;
    firstPitch?: string;
    weather?: string;
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
  factors?: Array<{
    name?: string;
    pct?: number;
    note?: string;
  }>;
};

type StaticSlatePayload = {
  date?: string;
  games?: StaticSlateGame[];
};

export type ModelSlateStatus = 'loaded' | 'missing' | 'error';

export type ModelSlateResult = {
  status: ModelSlateStatus;
  predictionsByGamePk: Map<number, StaticSlateGame>;
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

function parseNumber(value?: string) {
  if (!value) return null;
  const cleaned = value.replace(/[^\d.-]/g, '');
  if (!cleaned) return null;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function teamLabel(game: MlbGame, side: 'away' | 'home') {
  return side === 'away' ? game.awayTeam.abbreviation : game.homeTeam.abbreviation;
}

function categoryForStat(stat: string) {
  const upper = stat.toUpperCase();
  if (upper.includes('SP')) return 'Pitching';
  if (upper.includes('R/GAME')) return 'Form';
  if (upper.includes('WIN') || upper.includes('RECORD')) return 'Season';
  return 'Model';
}

function statDisplayName(stat: string) {
  const upper = stat.toUpperCase();
  if (upper.includes('R/GAME')) return 'Recent run production';
  if (upper.includes('SP ERA')) return 'Starter run prevention';
  if (upper.includes('SP WHIP')) return 'Starter traffic prevention';
  if (upper.includes('SP K/9')) return 'Starter strikeout profile';
  if (upper.includes('WIN PCT')) return 'Season win percentage';
  return stat;
}

function buildDifferentiatingFactors(game: MlbGame, staticGame: StaticSlateGame): ModelFactor[] {
  const rows = staticGame.stats ?? [];
  const scoredFactors: Array<{ score: number; factor: ModelFactor }> = [];

  rows.forEach((row) => {
      const stat = row.stat ?? '';
      const away = parseNumber(row.away);
      const home = parseNumber(row.home);
      if (away === null || home === null) return;

      const lowerIsBetter = /ERA|WHIP/i.test(stat);
      const awayBetter = lowerIsBetter ? away < home : away > home;
      const betterSide: 'away' | 'home' = awayBetter ? 'away' : 'home';
      const diff = Math.abs(away - home);
      const scale = /WIN PCT/i.test(stat)
        ? 0.08
        : /WHIP/i.test(stat)
          ? 0.12
          : /ERA/i.test(stat)
            ? 0.65
            : /K\/9/i.test(stat)
              ? 1.7
              : /R\/GAME/i.test(stat)
                ? 1.2
                : 1;
      const score = diff / scale;
      const betterTeam = teamLabel(game, betterSide);
      const worseTeam = teamLabel(game, betterSide === 'away' ? 'home' : 'away');
      const direction = betterSide === 'home' ? 'Positive' : 'Negative';
      const name = statDisplayName(stat);

      scoredFactors.push({
        score,
        factor: {
          name,
          description: `${betterTeam} owns the largest matchup gap in ${name.toLowerCase()}: ${betterSide === 'away' ? row.away : row.home} vs ${betterSide === 'away' ? row.home : row.away} for ${worseTeam}.`,
          impact: Math.round(Math.min(92, Math.max(44, 48 + score * 18))),
          direction,
          category: categoryForStat(stat),
          exampleSignal: `${row.away} vs ${row.home}`,
          affects: /R\/GAME|ERA|WHIP|K\/9/i.test(stat)
            ? ['Win probability', 'Projected score', 'Confidence']
            : ['Win probability', 'Confidence'],
        },
      });
    });

  const factors = scoredFactors
    .sort((a, b) => b.score - a.score)
    .map((item) => item.factor);

  if (factors.length) return factors.slice(0, 3);

  return factorsFromDrivers(staticGame.prediction?.drivers ?? []);
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

    return { status: 'loaded', predictionsByGamePk };
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
  const topFactors = buildDifferentiatingFactors(game, staticGame);

  return {
    id: game.id,
    gamePk: game.gamePk,
    date: game.date,
    time: game.timeET,
    venue: staticGame.game?.venue ?? game.venue,
    status: game.status,
    weather: staticGame.game?.weather ?? 'Weather from trained slate',
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
      'Weather, scratches, and market movement can narrow the edge.',
    ],
    explanation: buildModelExplanation(game, staticGame, confidenceLabel, topFactors),
    predictionSource: 'model',
    pitcherDetails: staticGame.pitchers,
  };
}
