import { Activity, BarChart3, Gauge, Home, LineChart, Settings, Table2, Trophy } from 'lucide-react';

export type NavKey = 'dashboard' | 'games' | 'game-detail' | 'accuracy' | 'factors' | 'backtest' | 'settings';

const navItems: Array<{ key: NavKey; label: string; icon: typeof Home }> = [
  { key: 'dashboard', label: 'Dashboard', icon: Home },
  { key: 'games', label: "Today's Games", icon: Table2 },
  { key: 'game-detail', label: 'Game Detail', icon: Gauge },
  { key: 'accuracy', label: 'Model Accuracy', icon: LineChart },
  { key: 'factors', label: 'Model Factors', icon: BarChart3 },
  { key: 'backtest', label: 'Backtest', icon: Trophy },
  { key: 'settings', label: 'Settings', icon: Settings },
];

export function Sidebar({ active, onNavigate }: { active: NavKey; onNavigate: (key: NavKey) => void }) {
  return (
    <aside className="hidden w-72 shrink-0 border-r border-white/10 bg-[#071426]/90 p-5 shadow-2xl shadow-slate-950/30 backdrop-blur-xl lg:block">
      <div className="mb-8 flex items-center gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-xl border border-teal-300/30 bg-teal-400/15 text-lg font-black text-teal-100">
          DF
        </div>
        <div>
          <p className="text-lg font-black tracking-tight text-white">Diamond Forecast</p>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">MLB analytics</p>
        </div>
      </div>

      <nav className="space-y-1">
        {navItems.map(({ key, label, icon: Icon }) => (
          <button
            className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm font-semibold transition ${
              active === key ? 'bg-white/12 text-white ring-1 ring-white/15' : 'text-slate-400 hover:bg-white/8 hover:text-slate-100'
            }`}
            key={key}
            onClick={() => onNavigate(key)}
          >
            <Icon size={18} />
            {label}
          </button>
        ))}
      </nav>

      <div className="mt-8 rounded-2xl border border-white/10 bg-white/8 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Model Confidence</span>
          <Activity size={16} className="text-teal-200" />
        </div>
        <div className="relative h-28">
          <div className="absolute inset-x-4 bottom-0 h-20 rounded-t-full border-8 border-b-0 border-slate-700" />
          <div className="absolute inset-x-4 bottom-0 h-20 rounded-t-full border-8 border-b-0 border-teal-300 [clip-path:inset(0_18%_0_0)]" />
          <div className="absolute inset-x-0 bottom-1 text-center">
            <strong className="text-3xl font-black text-white">71%</strong>
            <p className="text-xs text-slate-400">Slate average</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
