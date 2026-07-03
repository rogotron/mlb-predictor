import type { GamePrediction } from '../data/mockModelData';
import { teamColors } from '../utils/teamColors';

export function WinProbabilityBar({ game, compact = false }: { game: GamePrediction; compact?: boolean }) {
  const away = teamColors(game.awayTeam.abbreviation);
  const home = teamColors(game.homeTeam.abbreviation);

  return (
    <div className={compact ? 'min-w-[180px]' : 'w-full'}>
      <div className="mb-2 flex items-center justify-between text-xs font-bold text-slate-700">
        <span style={{ color: away.primary }}>{game.awayTeam.abbreviation} {game.awayWinProbability}%</span>
        <span style={{ color: home.primary }}>{game.homeTeam.abbreviation} {game.homeWinProbability}%</span>
      </div>
      <div className="flex h-2.5 overflow-hidden rounded-full bg-slate-200/65 ring-1 ring-slate-200/90">
        <div
          style={{
            width: `${game.awayWinProbability}%`,
            background: `linear-gradient(90deg, ${away.primary}, ${away.secondary})`,
            opacity: 0.86,
          }}
        />
        <div
          style={{
            width: `${game.homeWinProbability}%`,
            background: `linear-gradient(90deg, ${home.secondary}, ${home.primary})`,
            opacity: 0.86,
          }}
        />
      </div>
    </div>
  );
}
