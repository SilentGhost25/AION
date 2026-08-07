import { useAIONHealth } from "@/hooks/useAIONHealth";

export function AIONStatusBar() {
  const { data, isLoading, isError } = useAIONHealth();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground px-4 py-1 border-b">
        <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
        Connecting to AION...
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex items-center gap-2 text-xs text-destructive px-4 py-1 border-b">
        <span className="h-2 w-2 rounded-full bg-red-500" />
        AION backend unreachable — start aion_api.py
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4 text-xs text-muted-foreground px-4 py-1 border-b bg-muted/30">
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full bg-green-500" />
        AION {data?.api_version}
      </span>
      <span>Model: <strong>{data?.active_model}</strong></span>
      <span>Ollama: <strong>{data?.services.ollama}</strong></span>
      <span>{data?.models_available} models available</span>
    </div>
  );
}

export default AIONStatusBar;
