import { useState, useEffect } from "react";
import { aion, HealthResponse } from "@/api/client";

export function useAIONHealth(pollIntervalMs: number = 10000) {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isError, setIsError] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;

    async function checkHealth() {
      try {
        const res = await aion.health();
        if (isMounted) {
          setData(res);
          setIsError(false);
          setIsLoading(false);
        }
      } catch {
        if (isMounted) {
          setIsError(true);
          setIsLoading(false);
        }
      }
    }

    checkHealth();
    const timer = setInterval(checkHealth, pollIntervalMs);

    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, [pollIntervalMs]);

  return { data, isLoading, isError };
}

export default useAIONHealth;
