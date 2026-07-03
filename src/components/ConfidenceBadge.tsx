import type { GamePrediction } from '../data/mockModelData';

const tones: Record<GamePrediction['confidenceLabel'], string> = {
  High: 'border-teal-600/25 bg-teal-600/12 text-teal-900',
  Medium: 'border-sky-700/20 bg-sky-500/12 text-sky-900',
  Low: 'border-slate-500/25 bg-slate-500/12 text-slate-700',
};

export function ConfidenceBadge({ label }: { label: GamePrediction['confidenceLabel'] }) {
  return (
    <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] ${tones[label]}`}>
      {label}
    </span>
  );
}
