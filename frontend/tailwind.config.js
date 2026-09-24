/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        // LokLens AI brand palette
        brand: {
          50:  "#eef4ff",
          100: "#dce8ff",
          200: "#bad4ff",
          300: "#84b4ff",
          400: "#4b8bfc",
          500: "#2563eb",
          600: "#1d4ed8",
          700: "#1e40af",
          800: "#1e3a8a",
          900: "#1e3270",
        },
        surface: {
          900: "#0d0f14",
          800: "#141720",
          700: "#1a1e2e",
          600: "#222840",
          500: "#2d3450",
        },
        verdict: {
          supported:   "#22c55e",
          likely:      "#84cc16",
          partial:     "#eab308",
          misleading:  "#f97316",
          unsupported: "#ef4444",
          contradicted:"#dc2626",
          insufficient:"#6b7280",
          authentic:   "#22c55e",
          manipulated: "#ef4444",
          ai:          "#a855f7",
          inconclusive:"#6b7280",
        },
      },
      animation: {
        "fade-in":   "fadeIn 0.4s ease-out",
        "slide-up":  "slideUp 0.5s ease-out",
        "pulse-slow":"pulse 3s infinite",
        "scan":      "scan 2s linear infinite",
      },
      keyframes: {
        fadeIn:  { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(20px)" }, to: { opacity: "1", transform: "translateY(0)" } },
        scan:    { from: { transform: "translateY(-100%)" }, to: { transform: "translateY(100vh)" } },
      },
    },
  },
  plugins: [],
};
