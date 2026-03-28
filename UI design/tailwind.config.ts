import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "system-ui", "sans-serif"]
      },
      colors: {
        ink: {
          900: "#0b1a2b",
          800: "#16253a",
          700: "#24364d",
          500: "#3b556f"
        },
        haze: {
          100: "#f3f6f9",
          200: "#e4ebf2",
          300: "#cdd9e3"
        },
        accent: {
          500: "#2c7a7b",
          400: "#38b2ac",
          300: "#5fc9c2"
        }
      },
      boxShadow: {
        card: "0 18px 45px -30px rgba(12, 24, 39, 0.45)"
      }
    }
  },
  plugins: []
};

export default config;
