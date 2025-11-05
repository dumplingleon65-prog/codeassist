"use client";
// App-wide providers.
// Why: wraps React Query and AlchemyAccountProvider so pages/components can access auth state/hooks,
// and so SSR-hydrated initial state is applied consistently.

import { config, queryClient } from "@/app/login/alchemy";
import { AlchemyClientState } from "@account-kit/core";
import { AlchemyAccountProvider } from "@account-kit/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { PropsWithChildren } from "react";

export function Providers({ children, initialState }: PropsWithChildren<{ initialState?: AlchemyClientState }>) {
  return (
    <QueryClientProvider client={queryClient}>
      <AlchemyAccountProvider config={config} queryClient={queryClient} initialState={initialState}>
        {children}
      </AlchemyAccountProvider>
    </QueryClientProvider>
  );
}

