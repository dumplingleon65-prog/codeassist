"use client";

import React from "react";
import dynamic from "next/dynamic";

const AppWrapper = dynamic(() => import("../AppWrapper"), { ssr: false });

export function ClientOnly() {
  return <AppWrapper />;
}
