import { useEffect, useState, useCallback } from "react";
import { VersionSelector } from "@/components/version-selector";
import { MetricsCards } from "@/components/metrics-cards";
import { MetricsChart } from "@/components/metrics-chart";
import { RunDetailDialog } from "@/components/run-detail-dialog";
import { Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue, 
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { getVersions } from "@/lib/api";
import {
  type VersionInfo,
  type MetricCategory,
  METRIC_CATEGORIES,
  versionLabel,
} from "@/types/evaluation";

export default function EvaluationDashboard() {
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [category, setCategory] = useState<MetricCategory>('hit');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogRunId, setDialogRunId] = useState<number | null>(null);
  const [dialogLabel, setDialogLabel] = useState("");
  const statusOptions = METRIC_CATEGORIES.map(x => ({ value: x.key, label: x.label }));

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getVersions();
      setVersions(data);
      // Auto-select all versions
      setSelectedIds(data.map((v) => v.latest_run_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch versions");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const selectedVersions = versions.filter((v) =>
    selectedIds.includes(v.latest_run_id)
  );

  const handleVersionClick = (v: VersionInfo) => {
    setDialogRunId(v.latest_run_id);
    setDialogLabel(versionLabel(v));
    setDialogOpen(true);
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* Error alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-6">
          <div className="h-10 w-64 bg-muted animate-pulse rounded-md" />
          <div className="grid grid-cols-4 gap-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-muted animate-pulse rounded-xl" />
            ))}
          </div>
          <div className="h-[400px] bg-muted animate-pulse rounded-xl" />
        </div>
      )}

      {/* Dashboard content */}
      {!loading && !error && (
        <>
          {/* Version selector */}
          <VersionSelector
            versions={versions}
            selectedIds={selectedIds}
            onChange={setSelectedIds}
          />

          {/* KPI Cards */}
          <MetricsCards versions={selectedVersions} />

          {/* Chart area */}
          <div className="border rounded-xl bg-card p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium">Metrics Comparison</h3>
              <Select
                items={statusOptions}
                value={category}
                onValueChange={(newValue) => {
                  if (newValue) setCategory(newValue as MetricCategory);
                }}
              >
                <SelectTrigger className="w-full max-w-48">
                  <SelectValue placeholder="Select metric category"/>
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Metric Category</SelectLabel>
                    {statusOptions.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
            <MetricsChart
              versions={selectedVersions}
              category={category}
              onVersionClick={handleVersionClick}
            />
          </div>
        </>
      )}

      {/* Empty state */}
      {!loading && !error && versions.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-muted-foreground gap-2">
          <div className="text-4xl">📊</div>
          <p className="text-sm">No evaluation runs found</p>
          <p className="text-xs">Run the RAG evaluator to populate data, then refresh this page.</p>
        </div>
      )}

      {/* Run detail dialog */}
      <RunDetailDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        runId={dialogRunId}
        versionLabel_={dialogLabel}
      />
    </div>
  );
}
