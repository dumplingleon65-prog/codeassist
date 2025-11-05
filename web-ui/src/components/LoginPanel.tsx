"use client";
import { useEffect, useState, useCallback } from "react";
import { useSigner, useSignerStatus, useUser, useAuthModal } from "@account-kit/react";
import { alchemy, gensynTestnet } from "@account-kit/infra";
import { createModularAccountV2, createModularAccountV2Client } from "@account-kit/smart-contracts";
import { buildDeferredActionDigest, PermissionType, installValidationActions, PermissionBuilder, deferralActions } from "@account-kit/smart-contracts/experimental";
import "./LoginPanel.css";

const DAY_IN_MILLISECONDS = 1000 * 60 * 60 * 24;

export function LoginPanel() {
  const signerStatus = useSignerStatus();
  const { openAuthModal } = useAuthModal();
  const signer = useSigner();
  const user = useUser();

  const [authError, setAuthError] = useState<string | null>(null);


  // Handle account setup after authentication
  const handleAll = useCallback(async () => {
    if (!user || !signer) return;

    const whoamiStamp = await signer.inner.stampWhoami();
    const resp = await fetch("/api/login/get-api-key", { method: "POST", body: JSON.stringify({ whoamiStamp }) });
    const { publicKey } = await resp.json();

    const transport = alchemy({ apiKey: process.env.NEXT_PUBLIC_ALCHEMY_API_KEY! });
    const account = await createModularAccountV2({ transport, chain: gensynTestnet, signer });
    const initCode = await account.getInitCode();

    const client = (
      await createModularAccountV2Client({
        signer,
        signerEntity: account.signerEntity,
        accountAddress: account.address,
        transport,
        chain: gensynTestnet,
      })
    ).extend(installValidationActions).extend(deferralActions);

    const { entityId, nonce } = await client.getEntityIdAndNonce({ isGlobalValidation: true });

    const { typedData, fullPreSignatureDeferredActionDigest } = await new PermissionBuilder({
      client,
      key: { publicKey, type: "secp256k1" },
      entityId,
      nonce,
      deadline: 62 * DAY_IN_MILLISECONDS,
    })
      .addPermission({ permission: { type: PermissionType.ROOT } })
      .compileDeferred();

    const deferredValidationSig = await client.account.signTypedData(typedData);

    const deferredActionDigest = buildDeferredActionDigest({
      fullPreSignatureDeferredActionDigest,
      sig: deferredValidationSig,
    });

    await fetch("/api/login/set-api-key-activated", {
      method: "POST",
      body: JSON.stringify({
        orgId: user.orgId,
        apiKey: publicKey,
        accountAddress: account.address,
        initCode,
        deferredActionDigest,
      }),
    });

  }, [signer, user]);

  useEffect(() => {
    if (signerStatus.status === "CONNECTED") {
      handleAll();
    }
  }, [signerStatus.status, handleAll]);

  // Auto-open login modal only after confirming a true disconnected state; wait briefly to avoid flicker during hydration
  useEffect(() => {
    if (signerStatus.status !== "DISCONNECTED") {
      return;
    }

    // Wait 500ms to avoid flicker during hydration
    const t = setTimeout(() => {
      if (signerStatus.status === "DISCONNECTED") {
        openAuthModal();
      }
    }, 500);
    return () => clearTimeout(t);
  }, [signerStatus.status, openAuthModal]);

  // Validate Alchemy API key on mount
  useEffect(() => {
    let cancelled = false;
    async function preflight() {
      setAuthError(null);
      const key = process.env.NEXT_PUBLIC_ALCHEMY_API_KEY;
      if (!key || key === "dummy") {
        if (!cancelled) {
          setAuthError("Missing/invalid Alchemy key. Set NEXT_PUBLIC_ALCHEMY_API_KEY to a valid Account Kit key.");
          console.error("Missing/invalid Alchemy key");
        }
        return;
      }
      try {
        const url = "https://api.g.alchemy.com/signer/v1/signer-config";
        const base: RequestInit = {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: "{}",
        };
        let res = await fetch(url, {
          ...base,
          headers: { ...(base.headers as Record<string, string>), Authorization: `Bearer ${key}` },
        });
        if (res.status === 401 || res.status === 403) {
          // Fallback to legacy header for certain keys
          res = await fetch(url, {
            ...base,
            headers: { ...(base.headers as Record<string, string>), "x-alchemy-api-key": key as string },
          });
        }
        if (!cancelled) {
          if (res.ok) {
            try {
              const cfg = await res.json();
              // TEMP: print signer-config once for debugging
              console.log("Alchemy signer-config:", cfg);
            } catch {}
          } else if (res.status === 401 || res.status === 403) {
            setAuthError("Alchemy API key unauthorized. Update NEXT_PUBLIC_ALCHEMY_API_KEY.");
            console.error("Alchemy API key unauthorized");
          }
        }
      } catch (err) {
        console.error("Error validating Alchemy key:", err);
      }
    }
    preflight();
    return () => {
      cancelled = true;
    };
  }, []);

  // Display error banner if there's an auth error
  if (authError) {
    return (
      <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[1001] max-w-md">
        <div className="bg-red-900/90 border border-red-500 text-red-100 px-4 py-3 rounded text-sm">
          {authError}
        </div>
      </div>
    );
  }

  // The modal is managed by Alchemy's useAuthModal hook
  return null;
}

