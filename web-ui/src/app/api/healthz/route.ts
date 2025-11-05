import { NextResponse } from "next/server";

export async function GET(): Promise<NextResponse<{ status: string }>> {
  return NextResponse.json({ status: "ready" }, { status: 200 });
}
