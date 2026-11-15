import type { ReactNode } from "react";

type SectionProps = {
  id?: string;
  eyebrow?: string;
  title?: string;
  subtitle?: string;
  children: ReactNode;
};

export function Section({ id, eyebrow, title, subtitle, children }: SectionProps) {
  return (
    <section id={id} className="mx-auto w-full max-w-6xl px-4 py-14 sm:px-6 lg:px-8">
      {(eyebrow || title || subtitle) && (
        <div className="mb-8 max-w-2xl">
          {eyebrow && (
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-500">
              {eyebrow}
            </p>
          )}
          {title && (
            <h2 className="mt-3 font-display text-3xl font-semibold text-ink-900 md:text-4xl">
              {title}
            </h2>
          )}
          {subtitle && <p className="mt-3 text-base text-ink-500">{subtitle}</p>}
        </div>
      )}
      {children}
    </section>
  );
}
