import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type Datum = {
  name: string;
  value: number;
};

type ProbabilityChartProps = {
  data: Datum[];
};

const barColors: Record<string, string> = {
  AD: "#2c7a7b",
  DLB: "#3b82f6",
  PSP: "#f59e0b"
};

export function ProbabilityChart({ data }: ProbabilityChartProps) {
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 20, left: 12, bottom: 8 }}
        >
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(value) => `${Math.round(value * 100)}%`}
            tickLine={false}
            axisLine={false}
          />
          <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} />
          <Tooltip
            formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
            cursor={{ fill: "rgba(44, 122, 123, 0.08)" }}
          />
          <Bar dataKey="value" radius={[8, 8, 8, 8]}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={barColors[entry.name] || "#2c7a7b"} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              formatter={(value: number) => `${Math.round(value * 100)}%`}
              className="fill-ink-700 text-xs font-semibold"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
