import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#09090b",
        panel: "#141417",
        line: "#27272a",
        emerald: { 450: "#19a974" },
      },
      boxShadow: { panel: "0 18px 40px rgb(0 0 0 / 0.24)" },
    },
  },
  plugins: [],
};

export default config;
