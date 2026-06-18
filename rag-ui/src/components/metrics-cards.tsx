import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { VersionInfo } from "@/types/evaluation";
import { versionLabel } from "@/types/evaluation";
import { TrendingUp, Trophy, BarChart3, Layers } from "lucide-react";

interface MetricsCardsProps {
  versions: VersionInfo[];
}

export function MetricsCards({ versions }: MetricsCardsProps) {
  const stats = useMemo(() => {
    if (versions.length === 0) return null;

    let bestHit5 = versions[0];
    let bestMrr = versions[0];
    let bestNdcg5 = versions[0];

    for (const v of versions) {
      if (v.metrics["hit@5"] > bestHit5.metrics["hit@5"]) bestHit5 = v;
      if (v.metrics.mrr > bestMrr.metrics.mrr) bestMrr = v;
      if ((v.metrics["ndcg@5"] ?? 0) > (bestNdcg5.metrics["ndcg@5"] ?? 0)) bestNdcg5 = v;
    }

    return { bestHit5, bestMrr, bestNdcg5 };
  }, [versions]);

  if (!stats) {
    return null;
  }

  const cards = [
    {
      icon: Trophy,
      label: "Best Hit@5",
      value: stats.bestHit5.metrics["hit@5"].toFixed(4),
      version: versionLabel(stats.bestHit5),
      color: "text-blue-500",
    },
    {
      icon: TrendingUp,
      label: "Best MRR",
      value: stats.bestMrr.metrics.mrr.toFixed(4),
      version: versionLabel(stats.bestMrr),
      color: "text-emerald-500",
    },
    {
      icon: BarChart3,
      label: "Best NDCG@5",
      value: (stats.bestNdcg5.metrics["ndcg@5"] ?? 0).toFixed(4),
      version: versionLabel(stats.bestNdcg5),
      color: "text-purple-500",
    },
    {
      icon: Layers,
      label: "Total Versions",
      value: String(versions.length),
      version: `${versions.reduce((s, v) => s + v.run_count, 0)} runs total`,
      color: "text-muted-foreground",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((c) => (
        <Card key={c.label} size="sm">
          <CardHeader className="pb-1">
            <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
              <c.icon className={`size-3.5 ${c.color}`} />
              {c.label}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold tabular-nums">{c.value}</div>
            <div className="text-[10px] text-muted-foreground truncate mt-0.5">{c.version}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
