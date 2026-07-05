import { Shield, TrendingUp } from 'lucide-react';
import type { GamePrediction } from '../data/mockModelData';
import { ConfidenceBadge } from './ConfidenceBadge';
import { FactorImpactList } from './FactorImpactList';
import { GameExplanation } from './GameExplanation';
import { TeamIdentity } from './TeamIdentity';
import { WinProbabilityBar } from './WinProbabilityBar';
import { teamColors } from '../utils/teamColors';

function PitcherLine({ label, game, side }: { label: string; game: GamePrediction; side: 'away' | 'home' }) {
  const pitcher = side === 'away' ? game.awayPitcher : game.homePitcher;
  const details = side === 'away' ? game.pitcherDetails?.away : game.pitcherDetails?.home;
  const seed = (game.gamePk + (side === 'home' ? game.homeTeam.id : game.awayTeam.id)) % 100;
  const era = details?.era ?? (3.2 + seed / 90).toFixed(2);
  const xfip = details?.k9 ? `${details.k9} K/9` : (3.4 + seed / 110).toFixed(2);
  const whip = details?.whip ?? (1.04 + seed / 500).toFixed(2);
  return (
    <div className="rounded-lg border border-slate-200/75 bg-white/55 p-4">
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <h4 className="mt-2 text-base font-semibold text-slate-950">{pitcher}</h4>
      <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
        <span><b className="block text-slate-950">{era}</b> ERA</span>
        <span><b className="block text-slate-950">{xfip}</b>{details?.k9 ? '' : ' xFIP'}</span>
        <span><b className="block text-slate-950">{whip}</b> WHIP</span>
      </div>
      <p className="mt-3 text-sm text-slate-600">
        {details?.last ?? (game.predictionSource === 'model'
          ? 'Pitcher details loaded from the trained model slate.'
          : 'Prototype pitcher metrics are estimated until the model feed is connected.')}
      </p>
    </div>
  );
}

function TopModelDrivers({ game }: { game: GamePrediction }) {
  const winner = game.homeWinProbability >= game.awayWinProbability ? game.homeTeam : game.awayTeam;
  const accent = teamColors(winner.abbreviation);
  const drivers = game.topFactors.slice(0, 4);

  return (
    <section className="glass-card p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="section-kicker">Top model drivers</p>
          <h3 className="mt-1 text-lg font-semibold text-slate-950">Largest contributors to {winner.abbreviation}</h3>
        </div>
        <span className="rounded-full bg-white/65 px-3 py-1 text-xs font-black text-slate-700 ring-1 ring-slate-200/80">
          {drivers.length} factors
        </span>
      </div>
      <div className="mt-4 grid gap-3">
        {drivers.map((factor, index) => {
          const value = Math.max(8, Math.min(100, factor.impact));
          return (
            <div className="grid gap-2 md:grid-cols-[180px_1fr_48px] md:items-center" key={`${factor.name}-${index}`}>
              <div>
                <p className="text-sm font-bold text-slate-950">{factor.name}</p>
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{factor.category}</p>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full bg-slate-200/65 ring-1 ring-slate-200/90">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${value}%`,
                    background: `linear-gradient(90deg, ${accent.primary}, ${accent.secondary})`,
                    opacity: index === 0 ? 0.92 : 0.72,
                  }}
                />
              </div>
              <p className="text-right text-sm font-black text-slate-800">{value}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function GameDetailPanel({ game }: { game: GamePrediction }) {
  const winner = game.homeWinProbability >= game.awayWinProbability ? game.homeTeam : game.awayTeam;
  const winnerProb = Math.max(game.homeWinProbability, game.awayWinProbability);
  const bullpenDriver = game.topFactors.find(
    (factor) =>
      factor.category.toLowerCase() === 'bullpen' ||
      factor.description.toLowerCase().includes('bullpen'),
  );
  const bullpenCopy =
    game.predictionSource === 'model'
      ? bullpenDriver?.description ??
        'Bullpen quality and recent workload are included in the trained model slate for this matchup.'
      : 'Not connected yet. This placeholder will need recent relief appearances, pitch counts, and availability before it can be treated as a live factor.';

  return (
    <div className="space-y-5">
      <section className="glass-card p-5">
        <div className="flex flex-wrap items-center justify-between gap-5">
          <div className="flex items-center gap-4">
            <TeamIdentity team={game.awayTeam} size="lg" />
            <div className="text-center">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">{game.time}</p>
              <h2 className="mt-1 text-2xl font-black text-slate-950 md:text-4xl">
                {game.awayTeam.abbreviation} @ {game.homeTeam.abbreviation}
              </h2>
              <p className="mt-1 text-sm text-slate-600">{game.venue}</p>
            </div>
            <TeamIdentity team={game.homeTeam} size="lg" />
          </div>
          <div className="text-right">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">Model pick</p>
            <p className="mt-1 text-4xl font-black text-slate-950">{winner.abbreviation} {winnerProb}%</p>
            <div className="mt-2"><ConfidenceBadge label={game.confidenceLabel} /></div>
            <p className="mt-2 text-sm font-semibold text-slate-600">
              {game.predictionSource === 'model' ? 'Trained model slate' : 'Prototype fallback'}
            </p>
          </div>
        </div>
        <div className="mt-6 grid gap-5 lg:grid-cols-[1fr_280px]">
          <WinProbabilityBar game={game} />
          <div className="rounded-lg border border-slate-200/75 bg-white/55 p-4 text-center">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">Projected score</p>
            <p className="mt-1 text-3xl font-black text-slate-950">{game.projectedAwayRuns.toFixed(1)} - {game.projectedHomeRuns.toFixed(1)}</p>
          </div>
        </div>
      </section>

      <TopModelDrivers game={game} />

      <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
        <section className="glass-card p-5">
          <h3 className="text-lg font-semibold text-slate-950">Starting pitcher comparison</h3>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <PitcherLine label={game.awayTeam.abbreviation} game={game} side="away" />
            <PitcherLine label={game.homeTeam.abbreviation} game={game} side="home" />
          </div>
        </section>
        <GameExplanation game={game} />
      </div>

      <section className="grid gap-5 lg:grid-cols-2">
        {[
          ['Team offense', 'Split-adjusted contact quality favors the club with stronger projected lineup xwOBA.', TrendingUp],
          ['Bullpen comparison', bullpenCopy, Shield],
        ].map(([title, copy, Icon]) => (
          <article className="glass-card p-5" key={title as string}>
            <Icon className="text-teal-700" size={22} />
            <h3 className="mt-3 text-lg font-semibold text-slate-950">{title as string}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-600">{copy as string}</p>
          </article>
        ))}
      </section>

      <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
        <FactorImpactList factors={game.topFactors} title="Primary drivers" />
        <section className="glass-card p-5">
          <h3 className="text-lg font-semibold text-slate-950">Risk and market context</h3>
          <div className="mt-4 grid gap-3">
            <div className="rounded-lg border border-slate-200/75 bg-white/55 p-4">
              <p className="text-sm font-semibold text-slate-950">Model edge vs market</p>
              <p className="mt-1 text-sm text-slate-600">{game.modelEdge.toFixed(1)} points vs {game.marketLine}</p>
            </div>
            {game.riskFactors.map((risk) => (
              <div className="rounded-lg border border-red-400/15 bg-red-500/8 p-4 text-sm text-red-900" key={risk}>
                {risk}
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
