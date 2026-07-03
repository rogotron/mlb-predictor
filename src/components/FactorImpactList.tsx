import type { ModelFactor } from '../data/mockModelData';

export function FactorImpactList({ factors, title }: { factors: ModelFactor[]; title: string }) {
  return (
    <section className="glass-card p-5">
      <h3 className="text-lg font-semibold text-slate-950">{title}</h3>
      <div className="mt-4 space-y-4">
        {factors.map((factor) => (
          <div key={factor.name} title={factor.description}>
            <div className="mb-2 flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-slate-950">{factor.name}</p>
                <p className="mt-0.5 text-xs leading-5 text-slate-600">{factor.description}</p>
              </div>
              <span className={factor.direction === 'Negative' ? 'text-red-700' : factor.direction === 'Positive' ? 'text-teal-700' : 'text-slate-600'}>
                {factor.impact}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-200/80">
              <div
                className={factor.direction === 'Negative' ? 'h-full bg-red-500/70' : factor.direction === 'Positive' ? 'h-full bg-teal-600/70' : 'h-full bg-slate-500/60'}
                style={{ width: `${factor.impact}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
