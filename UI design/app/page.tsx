"use client";

import { useEffect, useState } from "react";
import { AboutPanel } from "../components/AboutPanel";
import { BatchPanel } from "../components/BatchPanel";
import { InferencePanel } from "../components/InferencePanel";
import { Section } from "../components/Section";
import { Tabs } from "../components/Tabs";
import { apiClient } from "../lib/api";

const tabs = [
  { id: "inference", label: "Inference", helper: "Single sample" },
  { id: "batch", label: "Batch", helper: "Multi-sample CSV" },
  { id: "about", label: "About", helper: "Evaluation details" }
];

export default function Home() {
  const [activeTab, setActiveTab] = useState("inference");
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      const ok = await apiClient.healthCheck();
      if (mounted) {
        setBackendHealthy(ok);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <main className="pb-16">
      <Section>
        <div className="section-glass rounded-3xl px-6 py-10 shadow-card sm:px-10">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-500">
                Inference Console
              </p>
              <h1 className="mt-3 font-display text-4xl font-semibold text-ink-900 md:text-5xl">
                NeuroFoldNet
              </h1>
              <p className="mt-3 max-w-2xl text-base text-ink-500">
                Replicate-aware ensemble classification of AD vs DLB vs PSP from quantitative tau
                polymorph features.
              </p>
            </div>
            <div className="rounded-2xl border border-haze-300 bg-white/80 px-5 py-4 text-sm text-ink-600">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-ink-500">
                Session status
              </p>
              {backendHealthy === null && (
                <>
                  <p className="mt-2">Checking backend status...</p>
                  <p className="text-xs text-ink-500">Waiting for health check.</p>
                </>
              )}
              {backendHealthy === true && (
                <>
                  <p className="mt-2">Connected to backend</p>
                  <p className="text-xs text-ink-500">Live model inference enabled.</p>
                </>
              )}
              {backendHealthy === false && (
                <>
                  <p className="mt-2">Backend offline</p>
                  <p className="text-xs text-ink-500">Start Flask server on port 5000.</p>
                </>
              )}
            </div>
          </div>
          <div className="mt-8">
            <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
          </div>
        </div>
      </Section>

      <Section>
        {activeTab === "inference" && <InferencePanel />}
        {activeTab === "batch" && <BatchPanel />}
        {activeTab === "about" && <AboutPanel />}
      </Section>

      <footer className="mx-auto w-full max-w-6xl px-4 pb-10 text-xs text-ink-500 sm:px-6 lg:px-8">
      </footer>
    </main>
  );
}

