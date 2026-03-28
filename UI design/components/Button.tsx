import type { ButtonHTMLAttributes } from "react";

const variants = {
  primary: "bg-ink-900 text-white shadow-lg shadow-ink-900/20 hover:bg-ink-800",
  secondary: "border border-ink-700/20 bg-white/70 text-ink-900 hover:border-ink-700/40"
};

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof variants;
};

export function Button({
  children,
  className = "",
  variant = "primary",
  ...props
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-full px-5 py-2.5 text-sm font-semibold transition ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
