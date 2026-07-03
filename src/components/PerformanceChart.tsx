import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type PerformancePoint = {
  day: string;
  winRate: number;
  brier: number;
  avgError: number;
  roi: number;
};

export function PerformanceChart({ data, compact = false }: { data: PerformancePoint[]; compact?: boolean }) {
  if (compact) {
    return (
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis dataKey="day" stroke="#94a3b8" tickLine={false} axisLine={false} />
            <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,.15)', borderRadius: 12 }} />
            <Line type="monotone" dataKey="winRate" stroke="#5eead4" strokeWidth={3} dot={false} name="Win rate" />
            <Line type="monotone" dataKey="roi" stroke="#93c5fd" strokeWidth={3} dot={false} name="ROI" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="winRate" x1="0" x2="0" y1="0" y2="1">
              <stop offset="5%" stopColor="#5eead4" stopOpacity={0.45} />
              <stop offset="95%" stopColor="#5eead4" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
          <XAxis dataKey="day" stroke="#94a3b8" tickLine={false} axisLine={false} />
          <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,.15)', borderRadius: 12 }} />
          <Area type="monotone" dataKey="winRate" stroke="#5eead4" strokeWidth={3} fill="url(#winRate)" name="Win rate" />
          <Line type="monotone" dataKey="roi" stroke="#fca5a5" strokeWidth={2} dot={false} name="ROI" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
