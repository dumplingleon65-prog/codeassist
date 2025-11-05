/**
 * GET /api/login/get-api-key-status — returns activation status and latest public key for an org.
 * Why: allows UI to poll/display status during/after the login flow.
 */
import { NextResponse } from "next/server";
import { getLatestKey } from "@/app/login/storage";

export const runtime = "nodejs";

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const orgId = url.searchParams.get("orgId");
    if (!orgId) throw new Error("Missing orgId");
    const key = await getLatestKey(orgId);
    return NextResponse.json({ activated: key?.activated === true, publicKey: key?.publicKey ?? null }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ error: String(err?.message || err) }, { status: 400 });
  }
}

