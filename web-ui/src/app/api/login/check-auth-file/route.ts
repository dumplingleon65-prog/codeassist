/**
 * POST /api/login/check-auth-file — checks if the user has a valid record in userKeyMap.json.
 * Why: allows the app to detect when persistent data has been deleted and clear stale cookies.
 *
 * Requires orgId in the request body to verify the specific user's record exists.
 */
import { NextResponse } from "next/server";
import { hasUserRecord } from "@/app/login/storage";
import { z } from "zod";

export const runtime = "nodejs";

const Body = z.object({
  orgId: z.string(),
});

export async function POST(req: Request) {
  try {
    const json = await req.json();
    const { orgId } = Body.parse(json);
    const exists = await hasUserRecord(orgId);
    return NextResponse.json({ exists }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ error: String(err?.message || err) }, { status: 500 });
  }
}

