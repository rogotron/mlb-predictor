import { AlertTriangle, CheckCircle2, CircleSlash, Database } from 'lucide-react';
import type { ModelSlateStatus, SlateProvenance, SlateSource } from '../services/modelSlate';

function formatRows(rows: number) {
  return rows.toLocaleString();
}

function SourceChip({ source }: { source: SlateSource }) {
  const notCollected = source.status === 'not_collected';
  const missing = source.status === 'missing' || source.status === 'error';
  const stale = source.stale && !notCollected;

  const tone = notCollected
    ? 'border-slate-300/60 bg-white/50 text-slate-500'
    : missing
      ? 'border-red-400/40 bg-red-500/10 text-red-900'
      : stale
        ? 'border-amber-500/40 bg-amber-200/50 text-amber-950'
        : 'border-teal-700/20 bg-teal-700/10 text-teal-900';

  const Icon = notCollected ? CircleSlash : missing ? AlertTriangle : stale ? AlertTriangle : CheckCircle2;

  const detail = notCollected
    ? 'not collected'
    : missing
      ? 'missing'
      : `${formatRows(source.rows)} rows${source.maxDate ? ` · thru ${source.maxDate}` : ''}${stale ? ' · stale' : ''}`;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${tone}`}
      title={`${source.label}: ${detail}`}
    >
      <Icon size={13} />
      <span className="font-bold">{source.label}</span>
      <span className="opacity-80">{detail}</span>
    </span>
  );
}

export function ProvenanceBar({
  provenance,
  modelSlateStatus,
  selectedDate,
}: {
  provenance?: SlateProvenance;
  modelSlateStatus: ModelSlateStatus;
  selectedDate: string;
}) {
  if (modelSlateStatus !== 'loaded' || !provenance) {
    return (
      <div className="mt-4 flex items-center gap-2 rounded-2xl border border-slate-300/50 bg-white/55 px-4 py-2.5 text-sm text-slate-600 backdrop-blur-md">
        <Database size={15} />
        No trained-model data provenance for {selectedDate}. Predictions on this date use the prototype fallback.
      </div>
    );
  }

  const sources = provenance.sources ?? [];
  const range = provenance.dateRange;
  const warn = provenance.anyStale || provenance.anyMissing;

  return (
    <div
      className={`mt-4 rounded-2xl border px-4 py-3 backdrop-blur-md ${
        warn ? 'border-amber-500/35 bg-amber-100/45' : 'border-slate-300/50 bg-white/60'
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
          <span className="font-bold text-slate-950">Data provenance</span>
          {provenance.dataAsOf ? (
            <span className="text-slate-600">
              as of <strong className="text-slate-800">{new Date(provenance.dataAsOf).toLocaleString()}</strong>
            </span>
          ) : null}
          {range?.start && range?.end ? (
            <span className="text-slate-600">
              range <strong className="text-slate-800">{range.start} → {range.end}</strong>
            </span>
          ) : null}
          {typeof provenance.gameCount === 'number' ? (
            <span className="text-slate-600">
              <strong className="text-slate-800">{provenance.gameCount}</strong> games
            </span>
          ) : null}
        </div>
        {warn ? (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-200/60 px-3 py-1 text-xs font-bold text-amber-950">
            <AlertTriangle size={13} />
            {provenance.anyMissing ? 'Source missing' : 'Source stale'}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-teal-700/20 bg-teal-700/12 px-3 py-1 text-xs font-bold text-teal-900">
            <CheckCircle2 size={13} />
            All sources current
          </span>
        )}
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {sources.map((source) => (
          <SourceChip key={source.key} source={source} />
        ))}
      </div>
    </div>
  );
}
