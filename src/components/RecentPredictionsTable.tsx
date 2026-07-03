import type { AccuracyRecord } from '../data/mockModelData';

export function RecentPredictionsTable({ records }: { records: AccuracyRecord[] }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-slate-300/45 bg-white/60 shadow-sm backdrop-blur-md">
      <div className="border-b border-slate-300/50 p-5">
        <p className="section-kicker">Prediction history</p>
        <h3 className="mt-1 text-lg font-semibold text-slate-950">Recent predictions and outcomes</h3>
      </div>
      <div className="grid grid-cols-[110px_1fr_1fr_110px_110px_1fr_120px_1fr_1fr_90px] gap-0 border-b border-slate-300/50 px-4 py-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-500 max-xl:hidden">
        <span>Date</span>
        <span>Matchup</span>
        <span>Predicted winner</span>
        <span>Win prob</span>
        <span>Confidence</span>
        <span>Actual winner</span>
        <span>Correct</span>
        <span>Projected score</span>
        <span>Actual score</span>
        <span>Error</span>
      </div>
      {records.map((record) => (
        <div
          className="grid gap-3 border-b border-slate-300/45 px-4 py-4 text-sm text-slate-700 last:border-b-0 xl:grid-cols-[110px_1fr_1fr_110px_110px_1fr_120px_1fr_1fr_90px] xl:items-center"
          key={record.gameId}
        >
          <span>{record.date}</span>
          <span className="font-semibold text-slate-950">{record.matchup}</span>
          <span>{record.predictedWinner}</span>
          <span>{record.predictedProbability}%</span>
          <span>{record.confidenceLabel}</span>
          <span>{record.actualWinner}</span>
          <span className={record.correct ? 'font-bold text-teal-800' : 'font-bold text-red-800'}>
            {record.correct ? 'Correct' : 'Incorrect'}
          </span>
          <span>{record.projectedAwayRuns.toFixed(1)} - {record.projectedHomeRuns.toFixed(1)}</span>
          <span>{record.actualAwayRuns} - {record.actualHomeRuns}</span>
          <span>{record.scoreError.toFixed(1)}</span>
        </div>
      ))}
    </section>
  );
}
