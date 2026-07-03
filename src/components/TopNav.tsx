import { CheckCircle2, Database, TriangleAlert } from 'lucide-react';

export function TopNav({
  selectedDate,
  onDateChange,
  onStepDate,
  lastUpdated,
  sourceLabel,
  usingFallback,
}: {
  selectedDate: string;
  onDateChange: (date: string) => void;
  onStepDate: (days: number) => void;
  lastUpdated: string;
  sourceLabel: string;
  usingFallback: boolean;
}) {
  return (
    <header className="glass-card flex flex-wrap items-center justify-between gap-4 px-5 py-4">
      <div>
        <h1 className="text-2xl font-black tracking-tight text-slate-950 md:text-3xl">Diamond Forecast</h1>
        <p className="mt-1 text-sm font-semibold text-slate-600">MLB Prediction Model</p>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="rounded-full border border-slate-300/60 bg-white/55 px-3 py-2 font-semibold text-slate-700">
          Model v4.2.1
        </span>
        <div className="inline-flex items-center rounded-full border border-slate-300/60 bg-white/55 p-1 font-semibold text-slate-700">
          <button className="rounded-full px-2 py-1 hover:bg-white/80" type="button" onClick={() => onStepDate(-1)}>
            Prev
          </button>
          <label className="px-2 py-1">
            <span className="sr-only">Selected date</span>
            <input
              className="w-[132px] bg-transparent text-slate-800 outline-none"
              type="date"
              value={selectedDate}
              onChange={(event) => onDateChange(event.target.value)}
            />
          </label>
          <button className="rounded-full px-2 py-1 hover:bg-white/80" type="button" onClick={() => onStepDate(1)}>
            Next
          </button>
        </div>
        <span className="rounded-full border border-slate-300/60 bg-white/55 px-3 py-2 font-semibold text-slate-700">
          Updated {lastUpdated}
        </span>
        <span
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 font-bold ${
            usingFallback
              ? 'border-amber-500/35 bg-amber-200/45 text-amber-900'
              : 'border-teal-700/20 bg-teal-700/12 text-teal-900'
          }`}
        >
          {usingFallback ? <TriangleAlert size={16} /> : <CheckCircle2 size={16} />}
          {sourceLabel}
        </span>
        <span className="inline-flex items-center gap-2 rounded-full border border-slate-300/60 bg-white/55 px-3 py-2 font-semibold text-slate-700">
          <Database size={16} />
          Schedule API
        </span>
        <div className="grid h-10 w-10 place-items-center rounded-full border border-slate-300/70 bg-slate-900 text-sm font-black text-white">
          DA
        </div>
      </div>
    </header>
  );
}
