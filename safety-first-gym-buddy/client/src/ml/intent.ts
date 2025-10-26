// Tiny online kNN classifier on short windows of pose features.
// It learns centroids from your demo videos automatically,
// then classifies webcam frames. No training dataset needed.

import { PoseMap } from "./features";

export type Label = "idle" | "squat" | "pushup" | "plank" | "deadbug" | "wallsit";

type Centroid = { sum: number[]; n: number; mean: number[] };

const FEAT_LEN = 7;

const centroids: Record<Label, Centroid> = {
  idle:     { sum: Array(FEAT_LEN).fill(0), n: 0, mean: Array(FEAT_LEN).fill(0) },
  squat:    { sum: Array(FEAT_LEN).fill(0), n: 0, mean: Array(FEAT_LEN).fill(0) },
  pushup:   { sum: Array(FEAT_LEN).fill(0), n: 0, mean: Array(FEAT_LEN).fill(0) },
  plank:    { sum: Array(FEAT_LEN).fill(0), n: 0, mean: Array(FEAT_LEN).fill(0) },
  deadbug:  { sum: Array(FEAT_LEN).fill(0), n: 0, mean: Array(FEAT_LEN).fill(0) },
  wallsit:  { sum: Array(FEAT_LEN).fill(0), n: 0, mean: Array(FEAT_LEN).fill(0) },
};

// seed idle so “sitting at desk” is recognized immediately
seed("idle", [90, 175, 160, 15, 15, 0.15, 0.5]); // body vertical, knees straight, little motion

function seed(label: Label, v: number[]) {
  const c = centroids[label];
  c.n += 1;
  for (let i = 0; i < FEAT_LEN; i++) c.sum[i] += v[i];
  c.mean = c.sum.map(x => x / c.n);
}

let training = false;
export function setTrainingMode(on: boolean) { training = on; }

// Feed a single frame into a centroid (used while playing demo videos)
export function learn(label: Label, v: number[]) {
  if (!training) return;
  seed(label, v);
}

// Compute per-frame features from a pose map and motion estimates
export function features(
  m: PoseMap,
  motionPx: number,
  stanceFrac: number,
  helpers: { bodyAngleAbs: number; knee: number; elbow: number; torsoAbs: number; lineDev: number }
): number[] {
  // 0 body angle abs (0 = horizontal, 90 = vertical)
  // 1 knee angle
  // 2 elbow angle
  // 3 torso tilt abs
  // 4 line deviation
  // 5 stance fraction ankles to shoulders
  // 6 motion pixels per frame
  return [
    clamp(helpers.bodyAngleAbs, 0, 180),
    clamp(helpers.knee, 0, 180),
    clamp(helpers.elbow, 0, 180),
    clamp(helpers.torsoAbs, 0, 90),
    clamp(helpers.lineDev, 0, 45),
    clamp(stanceFrac, 0, 1),
    clamp(motionPx / 20, 0, 2), // scaled motion
  ];
}

export function classify(v: number[]): { label: Label; conf: number; margin: number } {
  // nearest centroid
  let best: Label = "idle";
  let bestD = Number.POSITIVE_INFINITY;
  let second = Number.POSITIVE_INFINITY;

  for (const label of Object.keys(centroids) as Label[]) {
    const mean = centroids[label].mean;
    if (!mean.some(x => x !== 0)) continue; // skip empty
    const d = l2(v, mean);
    if (d < bestD) { second = bestD; bestD = d; best = label; }
    else if (d < second) { second = d; }
  }
  if (!isFinite(bestD)) return { label: "idle", conf: 0, margin: 0 };

  // confidence from distance ratio
  const margin = Math.max(0, second - bestD);
  const conf = 1 / (1 + bestD); // crude but effective
  return { label: best, conf, margin };
}

function l2(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < FEAT_LEN; i++) {
    const d = (a[i] - b[i]);
    s += d * d;
  }
  return s;
}

function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
