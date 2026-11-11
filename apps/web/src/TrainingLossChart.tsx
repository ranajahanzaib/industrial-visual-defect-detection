import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Epoch = {
  epoch: number;
  train_loss: number;
  calibration_loss: number;
};

export default function TrainingLossChart({ data }: { data: Epoch[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid stroke="#2b352e" />
        <XAxis dataKey="epoch" stroke="#8d998f" />
        <YAxis stroke="#8d998f" />
        <Tooltip />
        <Line dataKey="train_loss" stroke="#a9e86b" dot={false} />
        <Line dataKey="calibration_loss" stroke="#e5a953" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
