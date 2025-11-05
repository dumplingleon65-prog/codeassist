/**
 * POST /api/login/set-api-key-activated — marks the last created key as active and records account info.
 * Why: activates the key after client has constructed Modular Account and produced a deferredActionDigest
 */ 
import { NextResponse } from "next/server";
import { z } from "zod";
import { setKeyActivated } from "@/app/login/storage";

const Body = z.object({
  orgId: z.string(),
  apiKey: z.string(),
  accountAddress: z.string(),
  initCode: z.string(),
  deferredActionDigest: z.string(),
});

export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const json = await req.json();
    const { orgId, apiKey, accountAddress, initCode, deferredActionDigest } = Body.parse(json);

    const updated = await setKeyActivated(orgId, apiKey, { accountAddress, initCode, deferredActionDigest });
    if (!updated) {
      return NextResponse.json({ error: "Key not found" }, { status: 404 });
    }
    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ error: String(err?.message || err) }, { status: 400 });
  }
}

