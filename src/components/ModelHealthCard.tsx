import { Database, RadioTower, ShieldCheck } from 'lucide-react';

const checks = [
  { label: 'Prediction slate', value: 'Loaded', icon: Database },
  { label: 'Lineup freshness', value: 'Current', icon: RadioTower },
  { label: 'Calibration drift', value: 'Normal', icon: ShieldCheck },
];

export function ModelHealthCard() {
  return (
    <section className="glass-card p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="section-kicker">Model health status</p>
          <h3 className="text-lg font-semibold text-slate-950">Stable</h3>
        </div>
        <span className="rounded-full bg-teal-700/12 px-3 py-1 text-sm font-bold text-teal-900">Fresh</span>
      </div>
      <div className="mt-4 grid gap-3">
        {checks.map(({ label, value, icon: Icon }) => (
          <div className="flex items-center justify-between rounded-xl border border-slate-300/50 bg-white/45 p-3" key={label}>
            <span className="flex items-center gap-2 text-sm text-slate-700">
              <Icon size={16} className="text-teal-700" />
              {label}
            </span>
            <strong className="text-sm text-slate-950">{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
