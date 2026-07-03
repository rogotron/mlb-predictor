export interface MlbGame {
  id: string;
  gamePk: number;
  date: string;
  timeET: string;
  venue: string;
  status: string;
  awayTeam: {
    id: number;
    name: string;
    abbreviation: string;
    record?: string;
    logoUrl?: string;
    helmetUrl?: string;
  };
  homeTeam: {
    id: number;
    name: string;
    abbreviation: string;
    record?: string;
    logoUrl?: string;
    helmetUrl?: string;
  };
  awayPitcher?: string;
  homePitcher?: string;
  awayScore?: number;
  homeScore?: number;
  isFinal: boolean;
}

type StatsApiTeam = {
  id: number;
  name: string;
  abbreviation?: string;
  teamCode?: string;
  fileCode?: string;
};

type StatsApiGameTeam = {
  team: StatsApiTeam;
  leagueRecord?: {
    wins?: number;
    losses?: number;
    pct?: string;
  };
  probablePitcher?: {
    fullName?: string;
  };
  score?: number;
};

type StatsApiGame = {
  gamePk: number;
  gameDate: string;
  officialDate?: string;
  gameNumber?: number;
  doubleHeader?: string;
  status?: {
    abstractGameState?: string;
    detailedState?: string;
    codedGameState?: string;
    statusCode?: string;
  };
  venue?: {
    name?: string;
  };
  teams: {
    away: StatsApiGameTeam;
    home: StatsApiGameTeam;
  };
};

type StatsApiScheduleResponse = {
  dates?: Array<{
    date: string;
    games?: StatsApiGame[];
  }>;
};

function formatEasternTime(isoDate: string) {
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    timeZone: 'America/New_York',
    timeZoneName: 'short',
  }).format(new Date(isoDate));
}

function formatRecord(team: StatsApiGameTeam) {
  const wins = team.leagueRecord?.wins;
  const losses = team.leagueRecord?.losses;
  if (typeof wins !== 'number' || typeof losses !== 'number') return undefined;
  return `${wins}-${losses}`;
}

function teamAbbreviation(team: StatsApiTeam) {
  return (team.abbreviation ?? team.teamCode ?? team.fileCode ?? team.name.slice(0, 3)).toUpperCase();
}

function teamAssetSlug(team: StatsApiTeam) {
  return (team.fileCode ?? teamAbbreviation(team)).toLowerCase();
}

function mapTeam(team: StatsApiGameTeam) {
  const slug = teamAssetSlug(team.team);
  return {
    id: team.team.id,
    name: team.team.name,
    abbreviation: teamAbbreviation(team.team),
    record: formatRecord(team),
    logoUrl: `/assets/teams/${slug}-logo.png`,
    helmetUrl: `/assets/teams/${slug}-helmet.png`,
  };
}

function mapGame(game: StatsApiGame, date: string): MlbGame {
  const detailedState = game.status?.detailedState ?? game.status?.abstractGameState ?? 'Scheduled';
  return {
    id: String(game.gamePk),
    gamePk: game.gamePk,
    date,
    timeET: formatEasternTime(game.gameDate),
    venue: game.venue?.name ?? 'Venue TBD',
    status: detailedState,
    awayTeam: mapTeam(game.teams.away),
    homeTeam: mapTeam(game.teams.home),
    awayPitcher: game.teams.away.probablePitcher?.fullName ?? 'TBD',
    homePitcher: game.teams.home.probablePitcher?.fullName ?? 'TBD',
    awayScore: game.teams.away.score,
    homeScore: game.teams.home.score,
    isFinal: detailedState.toLowerCase().includes('final'),
  };
}

function shouldIncludeGame(game: StatsApiGame, requestedDate: string | null) {
  if (requestedDate && game.officialDate && game.officialDate !== requestedDate) return false;

  const statusText = [
    game.status?.abstractGameState,
    game.status?.detailedState,
    game.status?.codedGameState,
    game.status?.statusCode,
  ].filter(Boolean).join(' ').toLowerCase();

  return !['postpon', 'cancel', 'suspend', 'forfeit'].some((token) => statusText.includes(token));
}

async function fetchSchedule(params: URLSearchParams): Promise<MlbGame[]> {
  const response = await fetch(`https://statsapi.mlb.com/api/v1/schedule?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`MLB schedule request failed with HTTP ${response.status}`);
  }

  const payload = (await response.json()) as StatsApiScheduleResponse;
  const requestedDate = params.get('date');
  const games = payload.dates?.flatMap((scheduleDate) =>
    (scheduleDate.games ?? [])
      .filter((game) => shouldIncludeGame(game, requestedDate))
      .map((game) => mapGame(game, game.officialDate ?? scheduleDate.date)),
  );

  return games ?? [];
}

export async function fetchMlbSchedule(date: string): Promise<MlbGame[]> {
  const params = new URLSearchParams({
    sportId: '1',
    date,
    hydrate: 'probablePitcher,team,venue',
  });
  return fetchSchedule(params);
}

export async function fetchMlbScheduleRange(startDate: string, endDate: string): Promise<MlbGame[]> {
  const params = new URLSearchParams({
    sportId: '1',
    startDate,
    endDate,
    hydrate: 'probablePitcher,team,venue',
  });
  return fetchSchedule(params);
}
