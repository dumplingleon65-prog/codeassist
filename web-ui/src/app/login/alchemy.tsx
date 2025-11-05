/**
 * Alchemy Account Kit configuration used across Next app
 * This centralizes chain + transport + SSR settings so both server (SSR) and client use the same config.
 * We enable SSR to hydrate the client from cookies and avoid auth flicker, and we use Alchemy infra for
 * RPC and bundler behavior during the login flow.
 */
import { AlchemyAccountsUIConfig, cookieStorage, createConfig } from "@account-kit/react";
import { alchemy, gensynTestnet } from "@account-kit/infra";
import { QueryClient } from "@tanstack/react-query";
import Image from "next/image";

// Custom header component for CodeAssist branding
const CustomHeader = () => (
  <div className="flex flex-col items-center gap-6 mb-3">
    {/* CodeAssist Logo */}
    <div className="flex items-center justify-center">
      <Image
        src="/logos/codeassist.svg"
        alt="CodeAssist Logo"
        width={205}
        height={43}
        priority
      />
    </div>
    {/* Header text */}
    <div className="text-center">
      <div className="text-sm tracking-[0.12em] font-semibold mb-1.5">
        SIGN IN TO CODEASSIST
      </div>
      <div className="text-[10px] opacity-75 text-pink-base leading-relaxed">
        If you've used another Gensyn product before, please sign in with the same email and login method.
      </div>
    </div>
  </div>
);

const uiConfig: AlchemyAccountsUIConfig = {
  illustrationStyle: "outline",
  modalBaseClassName: "codeassist-auth-modal",
  auth: {
    sections: [
      [{ type: "email" }],
      [
        { type: "social", authProviderId: "google", mode: "popup" },
      ],
    ],
    addPasskeyOnSignup: false,
    header: <CustomHeader />,
    hideSignInText: true, // Hide default "Sign in" text since we have custom header
  },
};

const ALCHEMY_KEY = process.env.NEXT_PUBLIC_ALCHEMY_API_KEY ?? "dummy";

export const config = createConfig(
  {
    transport: alchemy({ apiKey: ALCHEMY_KEY }),
    chain: gensynTestnet,
    ssr: true,
    storage: cookieStorage,
    enablePopupOauth: true,
    sessionConfig: {
      expirationTimeMs: 1000 * 60 * 60 * 24 * 30, // 30 days
    },
  },
  uiConfig,
);

export const queryClient = new QueryClient();

