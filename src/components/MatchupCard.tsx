import { MapPin } from 'lucide-react';
import type { GamePrediction } from '../data/mockModelData';
import { ConfidenceBadge } from './ConfidenceBadge';
import { TeamIdentity } from './TeamIdentity';
import { WinProbabilityBar } from './WinProbabilityBar';
import { teamColors } from '../utils/teamColors';

export function MatchupCard({ game, onView }: { game: GamePrediction; onView: (game: GamePrediction) => void }) {
  const winner = game.homeWinProbability >= game.awayWinProbability ? game.homeTeam : game.awayTeam;
  const away = teamColors(game.awayTeam.abbreviation);
  const home = teamColors(game.homeTeam.abbreviation);
  const winnerAccent = teamColors(winner.abbreviation);

  return (
    <article
      className="glass-card overflow-hidden p-4"
      style={{
        borderColor: `${winnerAccent.primary}24`,
        background: `linear-gradient(135deg, ${away.primary}0f 0%, rgba(255,255,255,0.78) 34%, rgba(255,255,255,0.74) 66%, ${home.primary}10 100%)`,
      }}
    >
      <div
        className="-mx-4 -mt-4 mb-4 h-1"
        style={{ background: `linear-gradient(90deg, ${away.primary}, ${home.primary})` }}
      />
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <TeamIdentity team={game.awayTeam} />
          <span className="text-sm font-black text-slate-400">@</span>
          <TeamIdentity team={game.homeTeam} />
        </div>
        <ConfidenceBadge label={game.confidenceLabel} />
      </div>
      <div className="mt-3">
        <span className={`rounded-full px-3 py-1 text-xs font-bold ${
          game.predictionSource === 'model'
            ? 'bg-teal-700/10 text-teal-900 ring-1 ring-teal-700/15'
            : 'bg-amber-200/45 text-amber-950 ring-1 ring-amber-500/20'
        }`}>
          {game.predictionSource === 'model' ? 'Trained model prediction' : 'Prototype fallback prediction'}
        </span>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {[game.awayTeam, game.homeTeam].map((team) => (
          <div
            className="rounded-lg border border-slate-200/75 bg-white/55 p-3"
            key={team.id}
            style={{ boxShadow: `inset 3px 0 0 ${teamColors(team.abbreviation).primary}` }}
          >
            <p className="text-sm font-bold text-slate-950">{team.name}</p>
            <p className="mt-1 text-xs font-semibold text-slate-500">{team.record ?? 'Record pending'}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 space-y-2 text-sm text-slate-700">
        <p><strong className="text-slate-950">SP:</strong> {game.awayPitcher} vs {game.homePitcher}</p>
        <p className="flex items-center gap-2"><MapPin size={15} /> {game.time} · {game.venue}</p>
      </div>

      <div className="mt-5">
        <WinProbabilityBar game={game} />
        <div className="mt-3 flex items-center justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">Projected score</p>
            <p className="mt-1 text-lg font-black text-slate-950">
              {game.awayTeam.abbreviation} {game.projectedAwayRuns.toFixed(1)} · {game.homeTeam.abbreviation} {game.projectedHomeRuns.toFixed(1)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">Model edge</p>
            <p className="mt-1 text-lg font-black text-teal-800">{game.modelEdge > 0 ? '+' : ''}{game.modelEdge.toFixed(1)} pts</p>
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-slate-200/75 bg-white/55 p-3">
        <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">Top drivers</p>
        <div className="mt-2 flex flex-wrap gap-2">
          {game.topFactors.slice(0, 3).map((factor, index) => (
            <span className="rounded-full bg-slate-900/5 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200/80" key={`${factor.name}-${index}`} title={factor.description}>
              {factor.name}
            </span>
          ))}
        </div>
      </div>

      <button
        className="mt-4 w-full rounded-lg px-4 py-3 text-sm font-black text-slate-950 ring-1 transition hover:bg-white/75"
        style={{ backgroundColor: `${winnerAccent.primary}14`, borderColor: `${winnerAccent.primary}22`, boxShadow: `inset 0 0 0 1px ${winnerAccent.primary}22` }}
        onClick={() => onView(game)}
      >
        View Game Analysis: {winner.abbreviation}
      </button>
    </article>
  );
}
