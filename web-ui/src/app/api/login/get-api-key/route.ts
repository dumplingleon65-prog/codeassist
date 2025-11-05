/**
 * POST /api/login/get-api-key — creates and stores a public API key for the org (whoami-based).
 * Why: splits key creation from activation so the client can prove possession and install ROOT permission
 * before we mark the key as active.
 */
import { NextResponse } from "next/server";
import { z } from "zod";
import { addKey } from "@/app/login/storage";
import { generatePrivateKey, privateKeyToAccount } from "viem/accounts";


const ALCHEMY_BASE_URL = "https://api.g.alchemy.com";

const Body = z.object({
  // Accept the full stamped request object produced by Account Kit
  whoamiStamp: z.any(),
});

export const runtime = "nodejs";

async function verifyWhoami(stamp: unknown): Promise<{ orgId: string }> {
  const res = await fetch(`${ALCHEMY_BASE_URL}/signer/v1/whoami`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.NEXT_PUBLIC_ALCHEMY_API_KEY}`,
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ stampedRequest: stamp }),
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Alchemy whoami failed: ${res.status} ${txt}`);
  }

  const data: any = await res.json();
  if (!data?.orgId) {
    throw new Error("Alchemy whoami response missing orgId");
  }
  return { orgId: data.orgId as string };
}

export async function POST(req: Request) {
  try {
    const json = await req.json();
    const { whoamiStamp } = Body.parse(json);
    const { orgId } = await verifyWhoami(whoamiStamp);

    // Generate keypair and persist the address as the "publicKey"
    // Note: Account Kit permission builder expects the address string here (see rl-swarm reference)
    const privateKey = generatePrivateKey();
    const account = privateKeyToAccount(privateKey);
    const addressAsPublicKey = account.address;

    await addKey(orgId, addressAsPublicKey, privateKey);

    return NextResponse.json({ publicKey: addressAsPublicKey }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ error: String(err?.message || err) }, { status: 400 });
  }
}

