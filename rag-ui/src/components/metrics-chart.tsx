import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { useIsDark } from "@/hooks/useIsDark";
import {
  type MetricCategory,
  type VersionInfo,
  METRIC_K_VALUES,
  getMetricValue,
  versionLabel,
} from "@/types/evaluation";

interface MetricsChartProps {
  versions: VersionInfo[];
  category: MetricCategory;
  onVersionClick?: (version: VersionInfo) => void;
}

const CATEGORY_COLORS: Record<MetricCategory, string[]> = {
  hit: ["#3b82f6", "#6366f1", "#8b5cf6"],
  mrr: ["#10b981"],
  precision: ["#f59e0b", "#f97316", "#ef4444"],
  recall: ["#06b6d4", "#0891b2", "#0e7490"],
  ndcg: ["#ec4899", "#d946ef", "#a855f7"],
};

export function MetricsChart({ versions, category, onVersionClick }: MetricsChartProps) {
  const isDark = useIsDark();

  const option: EChartsOption = useMemo(() => {
    if (versions.length === 0) return {};

    const labels = versions.map((v) => versionLabel(v));
    const isMmr = category === "mrr";
    const seriesNames = isMmr ? ["MRR"] : METRIC_K_VALUES.map((k) => `${category}@${k}`);
    const colors = CATEGORY_COLORS[category];

    const series = seriesNames.map((name, idx) => ({
      name,
      type: "bar" as const,
      data: versions.map((v) => {
        const k = isMmr ? undefined : METRIC_K_VALUES[idx];
        return Number(getMetricValue(v.metrics, category, k).toFixed(4));
      }),
      itemStyle: {
        color: colors[idx % colors.length],
        borderRadius: [4, 4, 0, 0],
      },
      emphasis: {
        itemStyle: { color: colors[idx % colors.length], shadowBlur: 8, shadowColor: "rgba(0,0,0,0.3)" },
      },
      barMaxWidth: 48,
    }));

    const textColor = isDark ? "#cbd5e1" : "#334155";
    const gridColor = isDark ? "#1e293b" : "#e2e8f0";

    return {
      tooltip: {
        trigger: "axis" as const,
        axisPointer: { type: "shadow" as const },
        valueFormatter: (value: unknown) =>
          typeof value === "number" ? value.toFixed(4) : String(value),
        backgroundColor: isDark ? "#1e293b" : "#fff",
        borderColor: isDark ? "#334155" : "#e2e8f0",
        textStyle: { color: textColor },
      },
      legend: {
        data: seriesNames,
        bottom: 0,
        textStyle: { color: textColor },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "15%",
        top: "8%",
        containLabel: true,
      },
      xAxis: {
        type: "category" as const,
        data: labels,
        axisLabel: {
          color: textColor,
          rotate: labels.length > 3 ? 30 : 0,
          fontSize: 11,
          interval: 0,
          overflow: "truncate",
          width: 100,
        },
        axisLine: { lineStyle: { color: gridColor } },
        axisTick: { alignWithLabel: true },
      },
      yAxis: {
        type: "value" as const,
        name: "Score",
        min: 0,
        max: 1,
        axisLabel: {
          color: textColor,
          formatter: (val: number) => val.toFixed(1),
        },
        splitLine: { lineStyle: { color: gridColor } },
        nameTextStyle: { color: textColor },
      },
      series,
    } satisfies EChartsOption;
  }, [versions, category, isDark]);

  if (versions.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        Select versions to visualize metrics
      </div>
    );
  }

  return (
    <ReactECharts
      option={option}
      style={{ height: "400px", width: "100%" }}
      theme={isDark ? "dark" : undefined}
      onEvents={
        onVersionClick
          ? {
              click: (params: unknown) => {
                const p = params as { dataIndex?: number };
                if (typeof p.dataIndex === "number" && p.dataIndex >= 0) {
                  onVersionClick(versions[p.dataIndex]);
                }
              },
            }
          : undefined
      }
    />
  );
}
