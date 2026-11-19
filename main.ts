// Trophy Tide — GTA VI week-one trophy forecast (gta6-trophy-rarity-forecast-01).
//
// A cohort-mix Weibull model, written fresh for this bounty (Node 22, standard
// library only, no network, deterministic). It reads every <id>.in in the input
// directory and writes /output/<id>.out for each case:
//
//   {"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}
//
// The model
// --------
// Release origin T0 is 2026-11-19T00:00:00Z. For each read, tau is the hours
// from T0 to the read's `at` (clamped to at least 0.05). Consecutive identical
// reads (same players_tracked and same rates) collapse into one distinct state,
// because Exophase refreshes a game's counts on its own schedule, hours apart,
// and a run of identical reads is one observation, not many.
//
//   1. Tracked-players growth: least squares of log N on log(tau + 1) over the
//      distinct states gives a slope d, clamped to [0, 2]; with fewer than two
//      distinct states the prior d = 0.7 is used.
//   2. Per trophy, the earn rate is a mix of cohorts. Players arrive with
//      N(tau) = A (tau + 1)^d, so the inflow at join time s is
//      n(s) = A d (s + 1)^(d - 1), and a player who joined at s has earned by
//      tau with probability F(tau - s) = 1 - exp(-a (tau - s)^p). The rate is
//          r(tau) = 1 - [A exp(-a tau^p) + int_0^tau n(s) exp(-a (tau - s)^p) ds] / N(tau),
//      the integral on a fixed 100-point trapezoid grid over s in [0, tau].
//      (A cancels; the rate depends only on d, a and p.) The A exp(-a tau^p)
//      term is the initial cohort N(0) = A, who joined at s = 0: without it
//      the inflow integral only accounts for A[(tau+1)^d - 1] of the
//      A(tau+1)^d tracked players, and the model rate could never fall
//      below (tau+1)^(-d) — 42% at tau = 2.5 h with d = 0.7 — no matter how
//      small a is.
//   3. Fit: the global shape p is chosen over the grid {0.3, 0.4, ..., 1.5}.
//      For each p and trophy, a is solved by bisection on log a in
//      [-12, ln 2.3] (a is capped at 2.3 so a trophy near 100% does not imply
//      brand-new players earn instantly), matching the model's rate at the
//      last distinct state to the observed last rate in logit space.
//      Observed rates are clamped to [0.01, 99.99] for fitting, except an
//      observed 0.00% counts as 0.005; a last observed rate <= 0.005 puts a
//      at the lower bound. The p minimising the total squared logit error of
//      the model over all distinct states wins (ties go to p = 0.7).
//   4. The median at the case target is the model rate at the target's tau
//      with the chosen p and each trophy's a.
//   5. Quantiles: sigma = sqrt(max(resid_var, 0.04) + (0.45 sqrt(days))^2
//      + 0.01), where resid_var is the mean squared logit residual of the
//      fitted model over the distinct states and days = horizon_minutes /
//      1440; q_tau = inv_logit(logit(median) + z_tau sigma), z =
//      (-1.645, -0.674, 0, 0.674, 1.645), each q clamped to [0.01, 99.99]
//      and rounded to 4 decimals.
//
// Why this beats persistence: on launch night every read is flat, so the
// persistence baseline says rates never move. But Exophase's first refresh
// (638 -> 3,868 players tracked) pushed every rate but the 100% one down:
// new players arrive faster than they earn. The cohort mix models exactly
// that dilution — the rate at the target is the balance of fresh arrivals
// against the Weibull earning curve — instead of freezing today's rates.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Release origin: the start of GTA VI's release day, 2026-11-19T00:00:00Z. */
const T0_MS = Date.parse('2026-11-19T00:00:00Z');
/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Grid of Weibull shapes tried when fitting. */
const P_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
/** log a is bisected inside this range; a is capped at 2.3. */
const LOG_A_MIN = -12;
const LOG_A_MAX = Math.log(2.3);
/** Trapezoid points for the cohort integral. */
const GRID_POINTS = 100;
/** Prior growth slope with fewer than two distinct states. */
const D_PRIOR = 0.7;

interface TrophyMeta {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: TrophyMeta[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Hours from the release origin to an ISO timestamp, at least 0.05. */
const tauOf = (at: string): number => Math.max(0.05, (Date.parse(at) - T0_MS) / 3_600_000);

/** Observed rate as used for fitting: 0.00% counts as 0.005, else clamped. */
const fitRate = (r: number): number => (r === 0 ? 0.005 : clampPct(r));

interface State {
  tau: number;
  players: number;
  rates: Array<number | null>;
}

/**
 * Collapse consecutive identical reads (same players_tracked and same rates)
 * into distinct states. A state's tau is the tau of the last read in its run:
 * the most recent time that rate was observed.
 */
function distinctStates(history: Read[]): State[] {
  const states: State[] = [];
  for (const read of history) {
    const tau = tauOf(read.at);
    const prev = states[states.length - 1];
    if (
      prev !== undefined &&
      prev.players === read.players_tracked &&
      prev.rates.length === read.rates.length &&
      prev.rates.every((r, i) => r === read.rates[i])
    ) {
      prev.tau = tau;
      continue;
    }
    states.push({ tau, players: read.players_tracked, rates: read.rates.slice() });
  }
  return states;
}

/** Least-squares slope of log N on log(tau + 1); the prior when it is undefined. */
function fitGrowth(states: State[]): number {
  if (states.length < 2) return D_PRIOR;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  const n = states.length;
  for (const s of states) {
    const x = Math.log(s.tau + 1);
    const y = Math.log(Math.max(1, s.players));
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const denom = n * sxx - sx * sx;
  if (denom <= 0) return D_PRIOR;
  const slope = (n * sxy - sx * sy) / denom;
  return Math.min(2, Math.max(0, slope));
}

/**
 * The cohort-mix model rate at tau (hours since release) for growth slope d,
 * Weibull shape p and rate a. The integral is a fixed 100-point trapezoid
 * over join times s in [0, tau]; the scale A cancels.
 */
function modelRate(tau: number, d: number, p: number, a: number): number {
  if (tau <= 0) return 0;
  if (d === 0) return 100 * (1 - Math.exp(-a * Math.pow(tau, p)));
  const n = GRID_POINTS;
  const h = tau / (n - 1);
  const denom = Math.pow(tau + 1, d);
  // Not-earned mass: the initial cohort (joined at s = 0) plus the inflow.
  // Only the inflow integral is scaled by the trapezoid step h.
  const initial = Math.exp(-a * Math.pow(tau, p));
  let inflowIntegral = 0;
  for (let i = 0; i < n; i++) {
    const s = i * h;
    const w = i === 0 || i === n - 1 ? 0.5 : 1;
    const inflow = d * Math.pow(s + 1, d - 1);
    const surv = Math.exp(-a * Math.pow(tau - s, p));
    inflowIntegral += w * inflow * surv;
  }
  inflowIntegral *= h;
  const rate = 1 - (initial + inflowIntegral) / denom;
  return 100 * Math.min(1, Math.max(0, rate));
}

/**
 * Solve for a (by bisection on log a) so the model rate at tauLast equals the
 * observed last rate in logit space. A last observed rate <= 0.005 puts a at
 * the lower bound.
 */
function solveA(tauLast: number, lastRate: number, d: number, p: number): number {
  if (lastRate <= 0.005) return Math.exp(LOG_A_MIN);
  const target = logit(lastRate);
  const rateAt = (logA: number): number => modelRate(tauLast, d, p, Math.exp(logA));
  const lo = logit(Math.max(rateAt(LOG_A_MIN), 0.01));
  const hi = logit(Math.min(rateAt(LOG_A_MAX), 99.99));
  if (target <= lo) return Math.exp(LOG_A_MIN);
  if (target >= hi) return Math.exp(LOG_A_MAX);
  let a = LOG_A_MIN;
  let b = LOG_A_MAX;
  for (let iter = 0; iter < 80; iter++) {
    const mid = (a + b) / 2;
    if (logit(rateAt(mid)) < target) a = mid;
    else b = mid;
  }
  return Math.exp((a + b) / 2);
}

/** Squared logit error of one trophy's fitted model over the distinct states. */
function trophyError(
  states: State[],
  trophyIndex: number,
  d: number,
  p: number,
  a: number,
): number {
  let err = 0;
  let count = 0;
  for (const s of states) {
    const r = s.rates[trophyIndex];
    if (r === null || r === undefined) continue;
    const diff = logit(modelRate(s.tau, d, p, a)) - logit(fitRate(r));
    err += diff * diff;
    count++;
  }
  return count === 0 ? 0 : err / count;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);

  const states = distinctStates(c.history);
  const d = fitGrowth(states);
  const lastState = states[states.length - 1];
  const tauTarget = tauOf(c.target);
  const days = c.horizon_minutes / 1440;

  // Fit the global shape p: for each p, solve a per trophy against the last
  // distinct state, then total the squared logit error over all states.
  // Ties go to p = 0.7.
  const errors = new Map<number, number>();
  const aByP = new Map<number, number[]>();
  for (const p of P_GRID) {
    const as = c.trophies.map((_, i) => {
      const last = lastState.rates[i];
      if (last === null || last === undefined) return Math.exp(LOG_A_MIN);
      return solveA(lastState.tau, fitRate(last), d, p);
    });
    aByP.set(p, as);
    let total = 0;
    for (let i = 0; i < c.trophies.length; i++) total += trophyError(states, i, d, p, as[i]);
    errors.set(p, total);
  }
  const bestErr = Math.min(...errors.values());
  let pBest = P_GRID[0];
  if (Math.abs((errors.get(0.7) ?? Infinity) - bestErr) <= 1e-12) {
    pBest = 0.7;
  } else {
    for (const p of P_GRID) {
      if ((errors.get(p) ?? Infinity) === bestErr) {
        pBest = p;
        break;
      }
    }
  }
  const aBest = aByP.get(pBest) ?? [];

  const trophies = c.trophies.map((t, i) => {
    const a = aBest[i] ?? Math.exp(LOG_A_MIN);
    const median = clampPct(modelRate(tauTarget, d, pBest, a));
    const residVar = trophyError(states, i, d, pBest, a);
    const sigma = Math.sqrt(Math.max(residVar, 0.04) + Math.pow(0.45 * Math.sqrt(days), 2) + 0.01);
    const l = logit(median);
    const q = Z.map((z) => round4(clampPct(invLogitPct(l + z * sigma))));
    // Rounding can only move a value by 0.00005; enforce non-decreasing.
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
