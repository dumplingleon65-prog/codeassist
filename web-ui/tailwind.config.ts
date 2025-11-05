import { withAccountKitUi, createColorSet } from "@account-kit/react/tailwind";

// CodeAssist color palette
const colors = {
  darkBg: {
    base: "#1a1414",
    1: "#2a2222",
    2: "#3a3232",
  },
  pink: {
    base: "#FAD7D1",
  },
  green: {
    base: "#B8E986",
  },
  border: {
    base: "#4a4242",
  },
};

// Wrap your existing tailwind config with 'withAccountKitUi'
export default withAccountKitUi(
  {
    content: [
      "./src/**/*.{js,ts,jsx,tsx,mdx}",
      "./app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
      extend: {
        colors: {
          "dark-bg-base": colors.darkBg.base,
          "dark-bg-1": colors.darkBg[1],
          "dark-bg-2": colors.darkBg[2],
          "pink-base": colors.pink.base,
          "green-base": colors.green.base,
          "border-base": colors.border.base,
        },
        backgroundImage: {
          "gradient-radial": "radial-gradient(circle at 50% 50%, var(--tw-gradient-stops))",
        },
      },
    },
  },
  {
    // Override Account Kit theme colors to match CodeAssist
    colors: {
      "btn-primary": createColorSet(colors.green.base, colors.green.base),
      "btn-auth": createColorSet(colors.green.base, colors.green.base),
      "fg-accent-brand": createColorSet(colors.pink.base, colors.pink.base),
      "bg-surface-default": createColorSet(colors.darkBg.base, colors.darkBg.base), // Modal background
      "fg-primary": createColorSet(colors.pink.base, colors.pink.base), // Primary text color
      "fg-secondary": createColorSet(colors.pink.base, colors.pink.base),
      "fg-tertiary": createColorSet(colors.pink.base, colors.pink.base),
    },
    borderRadius: "sm", // Slightly rounded corners instead of sharp
  }
);

