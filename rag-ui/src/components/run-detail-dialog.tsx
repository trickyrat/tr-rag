import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { RunDetail, QueryResultItem } from "@/types/evaluation";
import { getRunDetail } from "@/lib/api";
import { cn } from "@/lib/utils";

interface RunDetailDialogProps {
  open: boolean;
  onClose: () => void;
  runId: number | null;
  versionLabel_?: string;
}

export function RunDetailDialog({ open, onClose, runId, versionLabel_ }: RunDetailDialogProps) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (open && runId !== null) {
      setLoading(true);
      setError(null);
      getRunDetail(runId)
        .then(setDetail)
        .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
        .finally(() => setLoading(false));
      setExpanded(new Set());
    }
  }, [open, runId]);

  const toggleExpand = (id: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="sm:max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-base">
            Run #{runId} {versionLabel_ ? `— ${versionLabel_}` : ""}
          </DialogTitle>
        </DialogHeader>

        {loading && (
          <div className="py-8 text-center text-muted-foreground text-sm">Loading...</div>
        )}

        {error && (
          <div className="py-8 text-center text-destructive text-sm">{error}</div>
        )}

        {detail && (
          <div className="space-y-4">
            {/* Config summary */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <ConfigKV label="Embedding" value={detail.config.embedding_model} />
              <ConfigKV label="Chunker" value={detail.config.chunker} />
              <ConfigKV label="Strategy" value={detail.config.retrieval_strategy} />
              <ConfigKV label="Top-K" value={String(detail.config.top_k)} />
              <ConfigKV label="Chunk Size" value={String(detail.config.chunk_size)} />
              <ConfigKV label="Cross Encoder" value={detail.config.use_cross_encoder ? "Yes" : "No"} />
              <ConfigKV label="Sparse" value={detail.config.use_sparse ? "Yes" : "No"} />
              <ConfigKV label="Elapsed" value={`${detail.elapsed_seconds.toFixed(1)}s`} />
            </div>

            {/* Per-query results */}
            <div>
              <h4 className="text-sm font-medium mb-2">
                Per-Query Results ({detail.query_results.length})
              </h4>
              <div className="space-y-1 max-h-96 overflow-y-auto border rounded-lg">
                {detail.query_results.map((qr) => (
                  <QueryRow
                    key={qr.id}
                    item={qr}
                    expanded={expanded.has(qr.id)}
                    onToggle={() => toggleExpand(qr.id)}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function ConfigKV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium truncate max-w-[140px] text-right">{value}</span>
    </div>
  );
}

function QueryRow({
  item,
  expanded,
  onToggle,
}: {
  item: QueryResultItem;
  expanded: boolean;
  onToggle: () => void;
}) {
  const hit = item.metrics?.["hit@5"] === 1;

  return (
    <div className="border-b last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-muted/50 transition-colors cursor-pointer text-left"
      >
        <span
          className={cn(
            "size-2 rounded-full shrink-0",
            hit ? "bg-emerald-500" : "bg-red-400"
          )}
        />
        <span className="truncate flex-1">{item.query}</span>
        <span className="text-muted-foreground tabular-nums shrink-0">
          {item.metrics
            ? `H@5:${item.metrics["hit@5"].toFixed(1)} MRR:${item.metrics.mrr.toFixed(2)}`
            : "-"}
        </span>
      </button>
      {expanded && (
        <div className="px-4 py-2 text-xs bg-muted/30 space-y-1">
          <div>
            <span className="text-muted-foreground">Retrieved:</span>{" "}
            {item.retrieved_ids.slice(0, 5).join(", ")}
            {item.retrieved_ids.length > 5 && ` +${item.retrieved_ids.length - 5} more`}
          </div>
          <div>
            <span className="text-muted-foreground">Relevant:</span>{" "}
            {item.relevant_ids.slice(0, 5).join(", ")}
            {item.relevant_ids.length > 5 && ` +${item.relevant_ids.length - 5} more`}
          </div>
          {item.category && (
            <div>
              <span className="text-muted-foreground">Category:</span> {item.category}
              {item.difficulty && ` / ${item.difficulty}`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
