import { cn } from "@/lib/utils";
import type { VersionInfo } from "@/types/evaluation";
import { versionLabel } from "@/types/evaluation";

interface VersionSelectorProps {
  versions: VersionInfo[];
  selectedIds: number[];
  onChange: (ids: number[]) => void;
}

export function VersionSelector({ versions, selectedIds, onChange }: VersionSelectorProps) {
  const allSelected = versions.length > 0 && selectedIds.length === versions.length;

  const toggle = (id: number) => {
    if (selectedIds.includes(id)) {
      onChange(selectedIds.filter((i) => i !== id));
    } else {
      onChange([...selectedIds, id]);
    }
  };

  const toggleAll = () => {
    if (allSelected) {
      onChange([]);
    } else {
      onChange(versions.map((v) => v.latest_run_id));
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Versions</span>
        <button
          onClick={toggleAll}
          className="text-xs text-primary hover:underline"
          type="button"
        >
          {allSelected ? "Clear all" : "Select all"}
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {versions.length === 0 && (
          <span className="text-sm text-muted-foreground">No versions found</span>
        )}
        {versions.map((v) => {
          const selected = selectedIds.includes(v.latest_run_id);
          return (
            <button
              key={v.latest_run_id}
              type="button"
              onClick={() => toggle(v.latest_run_id)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors cursor-pointer",
                selected
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground"
              )}
            >
              {versionLabel(v)}
              <span className={cn(
                "text-[10px] tabular-nums",
                selected ? "text-primary/70" : "text-muted-foreground/50"
              )}>
                (Run #{v.latest_run_id})
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
