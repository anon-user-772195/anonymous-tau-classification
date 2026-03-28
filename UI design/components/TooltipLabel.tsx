import type { ReactNode } from "react";

type TooltipLabelProps = {
  label: string;
  tooltip: string;
  htmlFor?: string;
  rightSlot?: ReactNode;
  isOpen?: boolean;
  onInfoToggle?: () => void;
};

export function TooltipLabel({
  label,
  tooltip,
  htmlFor,
  rightSlot,
  isOpen,
  onInfoToggle
}: TooltipLabelProps) {
  return (
    <div className="flex items-center justify-between gap-2">
      <label htmlFor={htmlFor} className="text-sm font-semibold text-ink-700">
        {label}
      </label>
      <div className="flex items-center gap-2">
        {rightSlot}
        {onInfoToggle ? (
          <button
            type="button"
            onClick={onInfoToggle}
            className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-haze-300 bg-white text-[11px] font-bold text-ink-500"
            title={tooltip}
            aria-label={tooltip}
            aria-pressed={isOpen}
          >
            i
          </button>
        ) : (
          <span
            className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-haze-300 bg-white text-[11px] font-bold text-ink-500"
            title={tooltip}
            aria-label={tooltip}
          >
            i
          </span>
        )}
      </div>
    </div>
  );
}
