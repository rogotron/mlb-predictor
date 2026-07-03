import type { LucideIcon } from 'lucide-react';

export function SummaryMetricCard({
  label,
  value,
  helper,
  icon: Icon,
  tone = 'neutral',
}: {
  label: string;
  value: string | number;
  helper?: string;
  icon?: LucideIcon;
  tone?: 'neutral' | 'positive' | 'negative';
}) {
  const toneClass =
    tone === 'positive' ? 'text-teal-900 bg-teal-600/12' : tone === 'negative' ? 'text-red-900 bg-red-500/12' : 'text-slate-800 bg-slate-900/8';

  return (
    <article className="glass-card p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">{label}</p>
        {Icon ? (
          <span className={`rounded-lg p-2 ${toneClass}`}>
            <Icon size={17} />
          </span>
        ) : null}
      </div>
      <strong className="mt-3 block text-2xl font-black tracking-tight text-slate-950">{value}</strong>
      {helper ? <span className="mt-1 block text-sm text-slate-600">{helper}</span> : null}
    </article>
  );
}
