"use client";
import { useEffect } from "react";
import { useLogout, useSignerStatus, useUser } from "@account-kit/react";

/**
 * Monitors the user's record in userKeyMap.json and automatically logs out
 * if the record is missing or empty.
 * Why: prevents stale cookie/login state when persistent data is deleted.
 *
 * Only checks when user is CONNECTED and we have their orgId to avoid
 * interrupting OTP input or other authentication steps.
 */
export function AuthFileMonitor() {
  const { logout } = useLogout();
  const signerStatus = useSignerStatus();
  const user = useUser();

  useEffect(() => {
    let cancelled = false;

    async function checkAuthFile() {
      // Only check if user is connected and we have their orgId
      // Don't check during INITIALIZING or AUTHENTICATING states to avoid interrupting login
      if (signerStatus.status !== "CONNECTED" || !user?.orgId) {
        return;
      }

      try {
        const response = await fetch("/api/login/check-auth-file", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ orgId: user.orgId }),
        });

        if (!response.ok) {
          console.error("Failed to check auth file:", response.statusText);
          return;
        }

        const { exists } = await response.json();

        if (!exists && !cancelled) {
          // Clear Alchemy cookies by logging out
          logout();
        }
      } catch (err) {
        console.error("Error checking auth file:", err);
      }
    }

    // Wait 3 seconds before first check to allow login flow to complete
    // This prevents race condition where we check before the record is created
    const initialTimeout = setTimeout(checkAuthFile, 3000);

    // Check periodically (every 30 seconds when connected)
    const interval = setInterval(checkAuthFile, 30000);

    return () => {
      cancelled = true;
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [logout, signerStatus.status, user?.orgId]);

  return null;
}

