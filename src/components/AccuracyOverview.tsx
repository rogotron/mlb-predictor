import { Info } from 'lucide-react';
import type { AccuracyRecord } from '../data/mockModelData';
import { SummaryMetricCard } from './SummaryMetricCard';

function pct(value: number) {
  return `${Math.round(value)}%`;
}

export function AccuracyOverview({ records }: { records: AccuracyRecord[] }) {
  if (!records.length) {
    return (
      <section className="glass-card p-5">
        <h3 className="text-lg font-semibold text-slate-950">No completed games available</h3>
        <p className="mt-2 text-sm text-slate-600">
          The selected window has no final games with scores available from the schedule feed.
        </p>
      </section>
    );
  }

  const accuracy = (records.filter((record) => record.correct).length / records.length) * 100;
  const favorites = records.filter((record) => record.predictedProbability >= 60);
  const underdogs = records.filter((record) => record.predictedProbability < 60);
  const favoriteAccuracy = (favorites.filter((record) => record.correct).length / Math.max(favorites.length, 1)) * 100;
  const underdogAccuracy = (underdogs.filter((record) => record.correct).length / Math.max(underdogs.length, 1)) * 100;
  const avgError = records.reduce((sum, record) => sum + record.scoreError, 0) / records.length;

  return (
    <section className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <SummaryMetricCard label="Overall accuracy" value={pct(accuracy)} helper={`${records.length} predictions`} tone="positive" />
        <SummaryMetricCard label="Favorites accuracy" value={pct(favoriteAccuracy)} helper={`${favorites.length} games`} />
        <SummaryMetricCard label="Underdogs accuracy" value={pct(underdogAccuracy)} helper={`${underdogs.length} games`} />
        <SummaryMetricCard label="Avg score error" value={avgError.toFixed(1)} helper="Runs per game" />
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[
          ['High bucket', '73%', 'Brier 0.214'],
          ['Medium bucket', '61%', 'Brier 0.229'],
          ['Low bucket', '54%', 'Brier 0.247'],
          ['Log loss', '0.668', 'Trailing sample'],
        ].map(([label, value, helper]) => (
          <div className="glass-card flex items-center justify-between p-4" key={label} title="Mock metric tooltip for analyst review.">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">{label}</p>
              <strong className="mt-2 block text-xl font-black text-slate-950">{value}</strong>
              <span className="text-sm text-slate-600">{helper}</span>
            </div>
            <Info className="text-slate-500" size={18} />
          </div>
        ))}
      </div>
    </section>
  );
}
