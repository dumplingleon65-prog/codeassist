import { Metadata } from "next";
import { headers } from "next/headers";
import { cookieToInitialState } from "@account-kit/core";
import { config } from "@/app/login/alchemy";
import "@/index.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "CodeAssist - AI Programming Assistant",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const initial = cookieToInitialState(
    config,
    (await headers()).get("cookie") ?? undefined,
  );
  return (
    <html lang="en">
      <body>
        <Providers initialState={initial}>
          <div id="root">{children}</div>
        </Providers>
      </body>
    </html>
  );
}
