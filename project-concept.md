# Prime Mover Tracker — Project Concept

*Updated: February 2026*

---

## What This Is

Prime Mover Tracker is a live intelligence system that identifies who is materially shaping global outcomes, maps how they relate to one another, and tracks how their influence moves week over week.

The core insight driving the design: influence is not a property of individuals in isolation — it flows through networks of competing and cooperating actors pursuing shared goals. A ranked list misses this. The project models influence as a dynamic system.

---

## What Has Been Built

The system is operational. Three interlocking components are running:

**1. LLM Council**
A multi-stage consensus pipeline queries four independent large language models (Claude, GPT, Grok, Gemini) on the same question: who are the most consequential actors right now, and why? Each model responds independently. Then each peer-reviews the others anonymously. A chairman model synthesizes all inputs into a final ranked list with justifications. This three-stage process reduces single-model bias and surfaces where models agree or diverge.

Two councils run in parallel:
- **Global Council** — 17 prime movers tracked across geopolitics, technology, finance, and energy.
- **US Domestic Council** — 12 US-based prime movers tracked across federal, corporate, judicial, and institutional domains.

**2. Superorganism Model**
The council output feeds into a Hebbian network model that treats the 17/12 individuals as neurons in a global power superorganism. Each actor is annotated with:
- Hemisphere (West / East / Bridge / Ancestral)
- Sector and organizational cell assemblies
- Phase sequences — the named goal-directed dynamics they participate in (e.g., AI supremacy race, dollar reserve contestation, tech-state integration)

Edges between actors are colored by valence: green for excitatory (cooperation), red for inhibitory (opposition), amber for both. The model makes competition and coalition structure visible at a glance.

**3. Weekly Briefing Pipeline**
An automated intelligence feed fetches per-person news via Perplexity (7-day window), pulls social signals on top movers via Grok/X, and synthesizes everything through Claude into structured JSON and Markdown reports. Each briefing includes per-person signal ratings (notable / quiet / concerning), phase sequence momentum assessments (accelerating / stable / decelerating), and an executive summary of the week's most significant moves.

---

## Current Tracked Individuals

**Global (17):**
Xi Jinping, Donald Trump, Elon Musk, Vladimir Putin, Narendra Modi, Jerome Powell, Jensen Huang, Sam Altman, Satya Nadella, Mohammed bin Salman, Tim Cook, Mark Zuckerberg, Ursula von der Leyen, Benjamin Netanyahu, Jamie Dimon, Larry Fink, Ali Khamenei

**US Domestic (12):**
Donald Trump, Elon Musk, John Roberts, Jerome Powell, Mike Johnson, Jensen Huang, Satya Nadella, Jamie Dimon, Mark Zuckerberg, Larry Fink, Sam Altman, Stephen Miller

---

## What the System Produces

- **Council result files** — Full JSON output with per-model responses, peer review scores, and chairman synthesis. One per run.
- **Superorganism model files** — Annotated JSON with canonical vocabulary, phase sequence mappings, and network edges.
- **Interactive visualizations** — Force-directed network graphs with hover tooltips. Global and US views combined in a single toggleable HTML file.
- **Weekly briefings** — Structured Markdown and JSON reports covering the most recent 7 days of activity per tracked individual.

---

## The Core Problem It Solves

News is fragmented by domain and outlet. Influence is cross-domain and fast-moving. Most people following geopolitics, markets, or technology are reading vertically — tech news, political news, financial news — when the most consequential signals cross those lines.

This system reads horizontally. It identifies the actors whose moves this week had downstream effects in multiple systems, and it maps those actors against each other so that coalitions, contests, and dependencies become legible.

---

## What Is Not Yet Built

The system is analytically strong but has no memory, no automation, and no user-facing layer.

**Historical depth** — Only a single briefing exists. The system needs to accumulate weeks of data before momentum tracking (who is rising, who is plateauing, who is losing leverage) becomes meaningful. Every run is currently a snapshot.

**Trend tracking** — Influence scores are re-derived from scratch each run. There is no persistent ledger that tracks score changes over time, compares current rankings to prior ones, or flags when a person crosses a threshold.

**Automation** — The weekly briefing requires a manual run. No scheduler generates reports automatically.

**Expanded coverage** — The weekly briefing currently tracks the US domestic 12. The global 17 are not yet fully wired into the briefing pipeline.

**User interface** — All outputs are static files (HTML, JSON, Markdown). There is no searchable dashboard, no watchlist feature, no person-page view, and no mobile-accessible format.

---

## The Meaningful Next Build

Before adding UI or expanding scope, the system needs a longitudinal spine: a persistent influence ledger that accumulates weekly snapshots and computes deltas. This is the feature that turns a weekly briefing into a real intelligence product.

The ledger would store each person's signal rating, phase sequence participation, and relative ranking per week. Each new briefing appends to this record. The system can then output:
- Who moved the most since last week (up/down)
- Which phase sequences are consistently accelerating
- Which actors have been quiet for multiple weeks in a row

Once this exists, the visualization gains a time dimension and the weekly briefing gains context.

Priority sequence from here:
1. Persistent influence ledger with weekly delta computation.
2. Global 17 fully integrated into weekly briefing pipeline.
3. Automated scheduling for weekly runs.
4. Timeline view in the network visualization.
5. Minimal web UI for browsing person pages and briefings.

---

## Guiding Principle

This is a map of power in motion, not a leaderboard. The goal is not to rank people but to make the structure of global influence legible — who is cooperating with whom, who is in opposition, which contests are heating up, and which actors are positioned to matter more or less in the weeks ahead.
