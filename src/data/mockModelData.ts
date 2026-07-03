import type { MlbGame } from '../services/mlbScheduleApi';

export interface ModelFactor {
  name: string;
  description: string;
  impact: number;
  direction: 'Positive' | 'Negative' | 'Neutral';
  category: string;
  exampleSignal: string;
  affects: Array<'Win probability' | 'Projected score' | 'Confidence'>;
}

export interface GamePrediction {
  id: string;
  gamePk: number;
  date: string;
  time: string;
  venue: string;
  status: string;
  weather: string;
  sourceGame: MlbGame;
  awayTeam: MlbGame['awayTeam'];
  homeTeam: MlbGame['homeTeam'];
  awayPitcher: string;
  homePitcher: string;
  awayWinProbability: number;
  homeWinProbability: number;
  projectedAwayRuns: number;
  projectedHomeRuns: number;
  confidenceScore: number;
  confidenceLabel: 'High' | 'Medium' | 'Low';
  modelEdge: number;
  marketLine: string;
  topFactors: ModelFactor[];
  riskFactors: string[];
  explanation: string;
  predictionSource: 'model' | 'prototype';
  pitcherDetails?: {
    away?: PitcherDetail;
    home?: PitcherDetail;
  };
}

export interface PitcherDetail {
  name: string;
  hand?: string;
  era?: string;
  whip?: string;
  k9?: string;
  bb9?: string;
  rec?: string;
  gs?: string;
  inn?: string;
  last?: string;
}

export interface AccuracyRecord {
  date: string;
  gameId: string;
  matchup: string;
  predictedWinner: string;
  predictedProbability: number;
  confidenceLabel: 'High' | 'Medium' | 'Low';
  actualWinner: string;
  correct: boolean;
  projectedAwayRuns: number;
  projectedHomeRuns: number;
  actualAwayRuns: number;
  actualHomeRuns: number;
  scoreError: number;
}

export interface PowerTeam {
  id: number | string;
  name: string;
  abbreviation: string;
  record?: string;
  logoUrl?: string;
  helmetUrl?: string;
  powerRating: number;
  powerRank: number;
  trend: 'up' | 'down' | 'flat';
}

const factor = (
  name: string,
  category: string,
  impact: number,
  direction: ModelFactor['direction'],
  description: string,
  exampleSignal: string,
  affects: ModelFactor['affects'],
): ModelFactor => ({ name, category, impact, direction, description, exampleSignal, affects });

export const modelFactors: ModelFactor[] = [
  factor('Starting pitcher edge', 'Pitching', 82, 'Positive', 'Compares probable starters using command, strikeout-minus-walk profile, and recent workload.', 'One starter projects with a cleaner K-BB profile and lower contact risk.', ['Win probability', 'Confidence']),
  factor('Bullpen strength', 'Bullpen', 68, 'Positive', 'Prototype placeholder for late-inning run prevention; live bullpen usage is not connected yet.', 'Needs recent relief appearances, pitch counts, and availability before this becomes live.', ['Win probability']),
  factor('Offensive split vs handedness', 'Offense', 74, 'Positive', 'Estimates lineup quality against the probable starter hand using rolling contact quality.', 'Projected lineup carries a stronger split against the opposing starter profile.', ['Win probability', 'Projected score']),
  factor('Recent form', 'Form', 61, 'Positive', 'Captures rolling run differential, plate discipline, and run-prevention stability.', 'Club has stronger two-week run differential and fewer late-inning collapses.', ['Win probability', 'Confidence']),
  factor('Home field / park factor', 'Context', 57, 'Neutral', 'Adjusts for venue, home-field advantage, scoring profile, and travel context.', 'Venue is near neutral but home field adds a small win-probability lift.', ['Win probability', 'Projected score']),
  factor('Weather impact', 'Weather', 42, 'Negative', 'Placeholder weather adjustment for temperature, wind, humidity, and precipitation risk.', 'Neutral placeholder until park-level weather feed is connected.', ['Projected score', 'Confidence']),
  factor('Injuries / lineup quality', 'Lineup', 66, 'Positive', 'Prototype placeholder for confirmed lineups, injuries, missing regulars, and defensive alignment.', 'Needs confirmed starting lineups and injury feed before this becomes live.', ['Win probability', 'Projected score', 'Confidence']),
  factor('Team offense', 'Offense', 70, 'Positive', 'Blends season-to-date production with recent contact quality and chase profile.', 'Top-half lineup quality is treated as above average in the mock layer.', ['Win probability', 'Projected score']),
  factor('Platoon splits', 'Offense', 64, 'Positive', 'Looks for matchup advantages created by batter handedness and starter profile.', 'Mock prediction gives a small boost when the starter matchup is favorable.', ['Win probability']),
  factor('Defense', 'Fielding', 48, 'Neutral', 'Uses defensive efficiency and run-prevention indicators to adjust balls in play.', 'Both teams are treated as near average until defensive feed is connected.', ['Win probability', 'Confidence']),
  factor('Rest and travel', 'Schedule', 53, 'Neutral', 'Accounts for off days, travel, bullpen fatigue, and getaway-game context.', 'Schedule context remains a light placeholder in the prototype.', ['Confidence']),
  factor('Market line comparison', 'Market', 76, 'Positive', 'Compares model probability with market-implied price to identify edge.', 'Prototype market edge is synthetic and should not be used for betting.', ['Confidence']),
];

function seededNumber(seed: string) {
  let hash = 0;
  for (let index = 0; index < seed.length; index += 1) {
    hash = (hash * 31 + seed.charCodeAt(index)) % 100000;
  }
  return hash;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function confidenceLabel(score: number): GamePrediction['confidenceLabel'] {
  if (score >= 74) return 'High';
  if (score >= 59) return 'Medium';
  return 'Low';
}

function winnerFor(game: GamePrediction) {
  return game.homeWinProbability >= game.awayWinProbability ? game.homeTeam : game.awayTeam;
}

function buildExplanation(game: Omit<GamePrediction, 'explanation'>) {
  const predictedWinner = game.homeWinProbability >= game.awayWinProbability ? game.homeTeam : game.awayTeam;
  const winProbability = Math.max(game.homeWinProbability, game.awayWinProbability);
  const mainFactor = game.topFactors[0];
  const supportFactor = game.topFactors[1] ?? modelFactors[1];
  const riskFactor = game.riskFactors[0] ?? 'the opposing lineup has enough variance to keep the game close';

  return [
    `The model gives the ${predictedWinner.name} a ${winProbability}% win probability, driven primarily by ${mainFactor.name.toLowerCase()}.`,
    `${predictedWinner.name} also benefits from ${supportFactor.name.toLowerCase()}, which improves the late-game projection if the matchup stays tight.`,
    `Weather and park effects are currently prototype placeholders, so the forecast keeps the run environment near neutral at ${game.projectedAwayRuns.toFixed(1)} to ${game.projectedHomeRuns.toFixed(1)}.`,
    `The main offsetting risk is ${riskFactor.toLowerCase()}, which keeps model confidence at ${game.confidenceLabel.toLowerCase()}.`,
  ].join(' ');
}

export function generateMockPredictionForGame(game: MlbGame): GamePrediction {
  const seed = seededNumber(`${game.gamePk}-${game.awayTeam.id}-${game.homeTeam.id}`);
  const homeLean = ((seed % 31) - 15) / 100;
  const starterKnownBonus = game.homePitcher !== 'TBD' && game.awayPitcher !== 'TBD' ? 0.03 : -0.02;
  const homeProbability = clamp(Math.round((0.53 + homeLean + starterKnownBonus) * 100), 38, 72);
  const awayProbability = 100 - homeProbability;
  const confidenceScore = clamp(Math.round(48 + Math.abs(homeProbability - 50) * 1.5 + (seed % 18)), 42, 86);
  const projectedTotal = 7.2 + (seed % 27) / 10;
  const homeRunShare = homeProbability / 100;
  const projectedHomeRuns = Number((projectedTotal * (0.47 + homeRunShare * 0.12)).toFixed(1));
  const projectedAwayRuns = Number((projectedTotal - projectedHomeRuns).toFixed(1));
  const topFactorIndexes = [seed % 5, (seed + 2) % 7, (seed + 5) % 8];
  const predictionBase = {
    id: game.id,
    gamePk: game.gamePk,
    date: game.date,
    time: game.timeET,
    venue: game.venue,
    status: game.status,
    weather: 'Weather feed pending; neutral park/weather placeholder',
    sourceGame: game,
    awayTeam: game.awayTeam,
    homeTeam: game.homeTeam,
    awayPitcher: game.awayPitcher ?? 'TBD',
    homePitcher: game.homePitcher ?? 'TBD',
    awayWinProbability: awayProbability,
    homeWinProbability: homeProbability,
    projectedAwayRuns,
    projectedHomeRuns,
    confidenceScore,
    confidenceLabel: confidenceLabel(confidenceScore),
    modelEdge: Number((((seed % 90) / 10) - 2.4).toFixed(1)),
    marketLine: 'Prototype line',
    topFactors: topFactorIndexes.map((index) => modelFactors[index]),
    riskFactors: [
      game.awayPitcher === 'TBD' || game.homePitcher === 'TBD'
        ? 'probable pitcher uncertainty'
        : 'bullpen usage and lineup confirmation are not yet connected',
      'prototype predictions are not connected to the trained model yet',
    ],
    predictionSource: 'prototype' as const,
  };
  const prediction = {
    ...predictionBase,
    explanation: buildExplanation(predictionBase),
  };

  return prediction;
}

export function generateAccuracyRecordForGame(game: MlbGame): AccuracyRecord | null {
  if (typeof game.awayScore !== 'number' || typeof game.homeScore !== 'number') {
    return null;
  }

  const prediction = generateMockPredictionForGame(game);
  const predictedWinner = predictionWinner(prediction);
  const actualWinner = game.homeScore >= game.awayScore ? game.homeTeam : game.awayTeam;

  return {
    date: game.date,
    gameId: String(game.gamePk),
    matchup: `${game.awayTeam.abbreviation} @ ${game.homeTeam.abbreviation}`,
    predictedWinner: predictedWinner.abbreviation,
    predictedProbability: Math.max(prediction.awayWinProbability, prediction.homeWinProbability),
    confidenceLabel: prediction.confidenceLabel,
    actualWinner: actualWinner.abbreviation,
    correct: predictedWinner.id === actualWinner.id,
    projectedAwayRuns: prediction.projectedAwayRuns,
    projectedHomeRuns: prediction.projectedHomeRuns,
    actualAwayRuns: game.awayScore,
    actualHomeRuns: game.homeScore,
    scoreError:
      Math.abs(prediction.projectedAwayRuns - game.awayScore) +
      Math.abs(prediction.projectedHomeRuns - game.homeScore),
  };
}

export const fallbackScheduleGames: MlbGame[] = [
  {
    id: 'fallback-1',
    gamePk: 900001,
    date: new Date().toISOString().slice(0, 10),
    timeET: '7:05 PM ET',
    venue: 'Yankee Stadium',
    status: 'Fallback sample',
    awayTeam: {
      id: 111,
      name: 'Boston Red Sox',
      abbreviation: 'BOS',
      record: '0-0',
      logoUrl: '/assets/teams/bos-logo.png',
      helmetUrl: '/assets/teams/bos-helmet.png',
    },
    homeTeam: {
      id: 147,
      name: 'New York Yankees',
      abbreviation: 'NYY',
      record: '0-0',
      logoUrl: '/assets/teams/nyy-logo.png',
      helmetUrl: '/assets/teams/nyy-helmet.png',
    },
    awayPitcher: 'TBD',
    homePitcher: 'TBD',
    isFinal: false,
  },
  {
    id: 'fallback-2',
    gamePk: 900002,
    date: new Date().toISOString().slice(0, 10),
    timeET: '8:10 PM ET',
    venue: 'Daikin Park',
    status: 'Fallback sample',
    awayTeam: {
      id: 136,
      name: 'Seattle Mariners',
      abbreviation: 'SEA',
      record: '0-0',
      logoUrl: '/assets/teams/sea-logo.png',
      helmetUrl: '/assets/teams/sea-helmet.png',
    },
    homeTeam: {
      id: 117,
      name: 'Houston Astros',
      abbreviation: 'HOU',
      record: '0-0',
      logoUrl: '/assets/teams/hou-logo.png',
      helmetUrl: '/assets/teams/hou-helmet.png',
    },
    awayPitcher: 'TBD',
    homePitcher: 'TBD',
    isFinal: false,
  },
];

export function buildPowerRankingsFromPredictions(predictions: GamePrediction[]): PowerTeam[] {
  const teams = new Map<number, PowerTeam>();
  predictions.forEach((game, index) => {
    [game.awayTeam, game.homeTeam].forEach((team, sideIndex) => {
      if (!teams.has(team.id)) {
        const ratingSeed = seededNumber(`${team.id}-${game.gamePk}`);
        teams.set(team.id, {
          ...team,
          powerRating: Number((142 + (ratingSeed % 180) / 10).toFixed(1)),
          powerRank: index * 2 + sideIndex + 1,
          trend: ratingSeed % 3 === 0 ? 'up' : ratingSeed % 3 === 1 ? 'down' : 'flat',
        });
      }
    });
  });

  return [...teams.values()]
    .sort((a, b) => b.powerRating - a.powerRating)
    .slice(0, 5)
    .map((team, index) => ({ ...team, powerRank: index + 1 }));
}

export function predictionWinner(game: GamePrediction) {
  return winnerFor(game);
}

export const performanceTrend = [
  { day: 'Day 1', winRate: 58, brier: 0.231, avgError: 3.2, roi: 1.8 },
  { day: 'Day 5', winRate: 61, brier: 0.224, avgError: 3.1, roi: 3.4 },
  { day: 'Day 9', winRate: 56, brier: 0.239, avgError: 3.5, roi: -1.1 },
  { day: 'Day 13', winRate: 63, brier: 0.218, avgError: 2.9, roi: 5.7 },
  { day: 'Day 17', winRate: 60, brier: 0.226, avgError: 3.0, roi: 4.2 },
  { day: 'Day 21', winRate: 64, brier: 0.216, avgError: 2.8, roi: 7.1 },
  { day: 'Day 25', winRate: 59, brier: 0.228, avgError: 3.3, roi: 3.6 },
  { day: 'Day 30', winRate: 62, brier: 0.221, avgError: 3.0, roi: 5.2 },
];

export const calibrationData = [
  { bucket: '50-60%', predicted: 55, actual: 53 },
  { bucket: '60-70%', predicted: 65, actual: 64 },
  { bucket: '70-80%', predicted: 75, actual: 72 },
  { bucket: '80%+', predicted: 84, actual: 81 },
];

const outcomes = [
  ['2026-05-23', 'NYY @ BOS', 'NYY', 61, 'Medium', 'NYY', true, 4.7, 4.0, 6, 3],
  ['2026-05-23', 'LAD @ ARI', 'LAD', 64, 'High', 'ARI', false, 5.2, 3.8, 4, 5],
  ['2026-05-22', 'ATL @ PHI', 'ATL', 58, 'Medium', 'ATL', true, 4.6, 4.1, 5, 4],
  ['2026-05-22', 'SEA @ HOU', 'HOU', 56, 'Low', 'HOU', true, 3.8, 4.2, 2, 3],
  ['2026-05-21', 'BAL @ NYY', 'NYY', 67, 'High', 'NYY', true, 3.7, 5.1, 2, 7],
  ['2026-05-20', 'SD @ LAD', 'LAD', 60, 'Medium', 'SD', false, 3.9, 4.8, 5, 4],
  ['2026-05-19', 'NYM @ ATL', 'ATL', 63, 'Medium', 'ATL', true, 4.0, 5.0, 3, 6],
  ['2026-05-18', 'PHI @ BAL', 'PHI', 57, 'Low', 'PHI', true, 4.4, 4.2, 5, 2],
  ['2026-05-17', 'BOS @ NYY', 'NYY', 59, 'Medium', 'BOS', false, 4.2, 4.7, 6, 5],
  ['2026-05-16', 'HOU @ SEA', 'SEA', 54, 'Low', 'SEA', true, 3.9, 4.0, 3, 4],
  ['2026-05-15', 'ARI @ LAD', 'LAD', 71, 'High', 'LAD', true, 3.6, 5.4, 2, 8],
  ['2026-05-14', 'ATL @ NYM', 'ATL', 62, 'Medium', 'NYM', false, 4.8, 4.1, 4, 5],
] as const;

export const accuracyRecords: AccuracyRecord[] = outcomes.map((row, index) => {
  const [
    date,
    matchup,
    predictedWinner,
    predictedProbability,
    confidenceLabel,
    actualWinner,
    correct,
    projectedAwayRuns,
    projectedHomeRuns,
    actualAwayRuns,
    actualHomeRuns,
  ] = row;

  return {
    date,
    gameId: `actual-${index}`,
    matchup,
    predictedWinner,
    predictedProbability,
    confidenceLabel,
    actualWinner,
    correct,
    projectedAwayRuns,
    projectedHomeRuns,
    actualAwayRuns,
    actualHomeRuns,
    scoreError: Math.abs(projectedAwayRuns - actualAwayRuns) + Math.abs(projectedHomeRuns - actualHomeRuns),
  };
});
