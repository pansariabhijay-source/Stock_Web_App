/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        /* ── Stitch "Institutional Precision" Design System ── */
        /* Surface hierarchy — deep grays, never pure black */
        "background":                "#131313",
        "surface":                   "#131313",
        "surface-dim":               "#131313",
        "surface-container-lowest":  "#0E0E0E",
        "surface-container-low":     "#1B1B1C",
        "surface-container":         "#202020",
        "surface-container-high":    "#2A2A2A",
        "surface-container-highest": "#353535",
        "surface-bright":            "#393939",
        "surface-variant":           "#353535",
        "surface-tint":              "#A8C8FF",

        /* Primary — Institutional Blue */
        "primary":             "#B5CFFF",
        "primary-container":   "#8AB4F8",
        "primary-fixed":       "#D5E3FF",
        "primary-fixed-dim":   "#A8C8FF",
        "on-primary":          "#003061",
        "on-primary-container":"#0D4582",
        "on-primary-fixed":    "#001B3C",
        "inverse-primary":     "#315F9D",

        /* Secondary — Subdued structure */
        "secondary":             "#C1C7CD",
        "secondary-container":   "#41474D",
        "secondary-fixed":       "#DDE3E9",
        "secondary-fixed-dim":   "#C1C7CD",
        "on-secondary":          "#2B3136",
        "on-secondary-container":"#B0B6BC",

        /* Tertiary — Subdued Emerald (gains) */
        "tertiary":             "#95DEA9",
        "tertiary-container":   "#7AC28F",
        "tertiary-fixed":       "#A9F3BC",
        "tertiary-fixed-dim":   "#8ED7A1",
        "on-tertiary":          "#00391B",
        "on-tertiary-container":"#005029",

        /* Error — Desaturated Coral (losses) */
        "error":             "#FFB4AB",
        "error-container":   "#93000A",
        "on-error":          "#690005",
        "on-error-container":"#FFDAD6",

        /* Content on surfaces */
        "on-surface":         "#E5E2E1",
        "on-surface-variant":  "#C3C6D1",
        "on-background":       "#E5E2E1",
        "inverse-surface":     "#E5E2E1",
        "inverse-on-surface":  "#303030",

        /* Outline */
        "outline":         "#8D919B",
        "outline-variant": "#424750",
      },
      fontFamily: {
        "headline": ["Inter", "system-ui", "sans-serif"],
        "body":     ["Inter", "system-ui", "sans-serif"],
        "label":    ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        "sm":      "8px",
        "DEFAULT": "12px",
        "md":      "12px",
        "lg":      "16px",
        "xl":      "24px",
        "2xl":     "24px",
        "full":    "9999px",
      },
      keyframes: {
        'scroll-ticker': {
          '0%':   { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-25%)' },
        },
      },
      animation: {
        'ticker': 'scroll-ticker 40s linear infinite',
      },
    },
  },
  plugins: [],
}
