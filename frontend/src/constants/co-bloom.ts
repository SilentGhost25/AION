// ─────────────────────────────────────────────────────────────────────────────
//  CO-BLOOM MAPPING RULES
//  These are the hard institutional rules for DSATM.
//  The Python backend MUST enforce the same mapping during generation.
//  Change here and in the backend together if rules change.
// ─────────────────────────────────────────────────────────────────────────────

import type { BloomLevel } from "@/types";

export interface CoBloomRule {
  co: string;
  bloom: BloomLevel[];
}

/** Module → CO + allowed Bloom levels. Enforced during generation. */
export const CO_BLOOM_RULES: Record<number, CoBloomRule> = {
  1: { co: "CO1", bloom: ["L1", "L2"] },
  2: { co: "CO2", bloom: ["L3"] },
  3: { co: "CO3", bloom: ["L4"] },
  4: { co: "CO3", bloom: ["L4"] },
  5: { co: "CO2", bloom: ["L3"] },
};

export interface BloomDefinition {
  label: string;
  color: string;
  verbs: string[];
  description: string;
}

/** Full Bloom's Taxonomy definitions — used in UI and AI prompts. */
export const BLOOM_LEVELS: Record<BloomLevel, BloomDefinition> = {
  L1: {
    label: "Remember",
    color: "bg-slate-100 text-slate-700 border-slate-200",
    verbs: ["Define", "List", "State", "Recall", "Identify", "Name", "Reproduce"],
    description: "Recall facts and basic concepts",
  },
  L2: {
    label: "Understand",
    color: "bg-blue-100 text-blue-700 border-blue-200",
    verbs: ["Explain", "Describe", "Summarize", "Interpret", "Classify", "Paraphrase"],
    description: "Explain ideas or concepts",
  },
  L3: {
    label: "Apply",
    color: "bg-emerald-100 text-emerald-700 border-emerald-200",
    verbs: ["Solve", "Demonstrate", "Use", "Implement", "Apply", "Calculate", "Execute"],
    description: "Use information in a new situation",
  },
  L4: {
    label: "Analyze",
    color: "bg-amber-100 text-amber-700 border-amber-200",
    verbs: ["Analyze", "Differentiate", "Compare", "Break down", "Examine", "Distinguish"],
    description: "Draw connections among ideas",
  },
  L5: {
    label: "Evaluate",
    color: "bg-orange-100 text-orange-700 border-orange-200",
    verbs: ["Justify", "Assess", "Critique", "Judge", "Evaluate", "Defend", "Argue"],
    description: "Justify a decision or course of action",
  },
  L6: {
    label: "Create",
    color: "bg-violet-100 text-violet-700 border-violet-200",
    verbs: ["Design", "Develop", "Construct", "Formulate", "Create", "Produce", "Build"],
    description: "Produce new or original work",
  },
};

/** Convenience array for all module CO-Bloom rules. */
export const MODULE_CO_BLOOM = Object.entries(CO_BLOOM_RULES).map(([mod, rule]) => ({
  module: Number(mod),
  co: rule.co,
  bloom: rule.bloom,
}));
