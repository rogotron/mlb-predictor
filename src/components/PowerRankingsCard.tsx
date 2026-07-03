import { ArrowDown, ArrowRight, ArrowUp } from 'lucide-react';
import type { PowerTeam } from '../data/mockModelData';
import { TeamIdentity } from './TeamIdentity';

export function PowerRankingsCard({ teams }: { teams: PowerTeam[] }) {
  return (
    <section className="glass-card p-5">
      <h3 className="text-lg font-semibold text-slate-950">Power rankings</h3>
      <div className="mt-4 space-y-3">
        {teams.map((team) => {
          const TrendIcon = team.trend === 'up' ? ArrowUp : team.trend === 'down' ? ArrowDown : ArrowRight;
          return (
            <div className="flex items-center gap-3 rounded-xl border border-slate-300/50 bg-white/45 p-3" key={team.id}>
              <span className="w-6 text-sm font-black text-slate-500">#{team.powerRank}</span>
              <TeamIdentity team={team} size="sm" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold text-slate-950">{team.name}</p>
                <p className="text-xs text-slate-500">Rating {team.powerRating.toFixed(1)}</p>
              </div>
              <TrendIcon className={team.trend === 'up' ? 'text-teal-700' : team.trend === 'down' ? 'text-red-700' : 'text-slate-500'} size={18} />
            </div>
          );
        })}
      </div>
    </section>
  );
}
