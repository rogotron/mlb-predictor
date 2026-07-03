const FALLBACK = {
  primary: '#0f766e',
  secondary: '#0f172a',
};

const TEAM_COLORS: Record<string, { primary: string; secondary: string }> = {
  ARI: { primary: '#a71930', secondary: '#e3d4ad' },
  ATL: { primary: '#ce1141', secondary: '#13274f' },
  BAL: { primary: '#df4601', secondary: '#000000' },
  BOS: { primary: '#bd3039', secondary: '#0c2340' },
  CHC: { primary: '#0e3386', secondary: '#cc3433' },
  CWS: { primary: '#27251f', secondary: '#c4ced4' },
  CIN: { primary: '#c6011f', secondary: '#000000' },
  CLE: { primary: '#e31937', secondary: '#0c2340' },
  COL: { primary: '#33006f', secondary: '#c4ced4' },
  DET: { primary: '#0c2340', secondary: '#fa4616' },
  HOU: { primary: '#eb6e1f', secondary: '#002d62' },
  KC: { primary: '#004687', secondary: '#bd9b60' },
  LAA: { primary: '#ba0021', secondary: '#003263' },
  LAD: { primary: '#005a9c', secondary: '#ef3e42' },
  MIA: { primary: '#00a3e0', secondary: '#ef3340' },
  MIL: { primary: '#ffc52f', secondary: '#12284b' },
  MIN: { primary: '#002b5c', secondary: '#d31145' },
  NYM: { primary: '#002d72', secondary: '#ff5910' },
  NYY: { primary: '#0c2340', secondary: '#c4ced4' },
  OAK: { primary: '#003831', secondary: '#efb21e' },
  PHI: { primary: '#e81828', secondary: '#002d72' },
  PIT: { primary: '#fdb827', secondary: '#27251f' },
  SD: { primary: '#2f241d', secondary: '#ffc425' },
  SEA: { primary: '#0c2c56', secondary: '#005c5c' },
  SF: { primary: '#fd5a1e', secondary: '#27251f' },
  STL: { primary: '#c41e3a', secondary: '#0c2340' },
  TB: { primary: '#092c5c', secondary: '#8fbce6' },
  TEX: { primary: '#003278', secondary: '#c0111f' },
  TOR: { primary: '#134a8e', secondary: '#1d2d5c' },
  WSH: { primary: '#ab0003', secondary: '#14225a' },
};

export function teamColors(abbreviation?: string) {
  if (!abbreviation) return FALLBACK;
  return TEAM_COLORS[abbreviation.toUpperCase()] ?? FALLBACK;
}
