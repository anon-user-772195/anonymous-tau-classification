import type { ReactNode } from "react";

type TabItem = {
  id: string;
  label: string;
  helper?: ReactNode;
};

type TabsProps = {
  tabs: TabItem[];
  activeTab: string;
  onChange: (tabId: string) => void;
};

export function Tabs({ tabs, activeTab, onChange }: TabsProps) {
  return (
    <div className="no-print rounded-2xl border border-haze-300 bg-white/80 p-2 shadow-card">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onChange(tab.id)}
              className={`flex-1 rounded-xl px-4 py-2 text-sm font-semibold transition sm:flex-none ${
                isActive
                  ? "bg-ink-900 text-white shadow-lg shadow-ink-900/20"
                  : "border border-transparent text-ink-600 hover:border-ink-700/20 hover:bg-haze-100"
              }`}
              aria-pressed={isActive}
            >
              <span className="block text-center sm:text-left">{tab.label}</span>
              {tab.helper && <span className="mt-1 block text-xs text-ink-500">{tab.helper}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}
