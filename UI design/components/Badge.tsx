import type { HTMLAttributes } from "react";

export function Badge({ children, className = "" }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={`inline-flex items-center rounded-full border border-haze-300 bg-white/80 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-ink-700 ${className}`}
    >
      {children}
    </span>
  );
}
