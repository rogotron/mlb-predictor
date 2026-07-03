import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

type CalibrationPoint = {
  bucket: string;
  predicted: number;
  actual: number;
};

export function CalibrationChart({ data }: { data: CalibrationPoint[] }) {
  return (
    <section className="glass-card p-5">
      <h3 className="mb-4 text-lg font-semibold text-slate-950">Calibration by probability bucket</h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis dataKey="bucket" stroke="#94a3b8" tickLine={false} axisLine={false} />
            <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,.15)', borderRadius: 12 }} />
            <Legend />
            <Bar dataKey="predicted" fill="#93c5fd" radius={[6, 6, 0, 0]} name="Predicted win rate" />
            <Bar dataKey="actual" fill="#5eead4" radius={[6, 6, 0, 0]} name="Actual win rate" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
