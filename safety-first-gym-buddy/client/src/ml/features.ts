// Keypoint typing
export type KP = { x: number; y: number; score?: number };
export type PoseMap = Record<string, KP | undefined>;

// Map array of keypoints to names (adds left/right where useful)
export function indexKeypoints(pose: { keypoints: KP[] }): PoseMap {
  const byName: Record<string, KP> = {};
  for (const p of (pose.keypoints as any[])) {
    const name = p.name || p.part || "";
    if (name) byName[name] = p;
  }
  const pick = (base: string) =>
    byName["left_" + base] || byName["left" + base] || byName[base] ||
    byName["right_" + base] || byName["right" + base];
  return {
    // single-side defaults (prefer left)
    hip: pick("hip"), knee: pick("knee"), ankle: pick("ankle"),
    shoulder: pick("shoulder"), elbow: pick("elbow"), wrist: pick("wrist"),
    head: byName["nose"] || pick("eye"),
    // explicit sides for feet/hips
    lHip: byName["left_hip"],  rHip: byName["right_hip"],
    lAnkle: byName["left_ankle"], rAnkle: byName["right_ankle"],
    lHeel: byName["left_heel"],   rHeel: byName["right_heel"],
    lToe: byName["left_foot_index"], rToe: byName["right_foot_index"],
    lShoulder: byName["left_shoulder"], rShoulder: byName["right_shoulder"],
  };
}

// Angles/distances
export function jointAngle(a?: KP, b?: KP, c?: KP): number {
  if (!a || !b || !c) return NaN;
  const abx = a.x - b.x, aby = a.y - b.y;
  const cbx = c.x - b.x, cby = c.y - b.y;
  const dot = abx * cbx + aby * cby;
  const mag1 = Math.hypot(abx, aby), mag2 = Math.hypot(cbx, cby);
  const cos = dot / (mag1 * mag2 + 1e-6);
  return Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
}
export function lineTiltFromVertical(a?: KP, b?: KP): number {
  if (!a || !b) return 0;
  const dy = a.y - b.y, dx = a.x - b.x;
  return Math.abs(Math.atan2(dx, dy) * 180 / Math.PI);
}
export function pointDeviationFromLine(a?: KP, b?: KP, p?: KP): number {
  if (!a || !b || !p) return 999;
  const dx = b.x - a.x, dy = b.y - a.y;
  const num = Math.abs(dx * (a.y - p.y) - dy * (a.x - p.x));
  const den = Math.hypot(dx, dy) + 1e-6;
  const dist = num / den;
  return Math.min(90, dist * 0.2);
}
export function dist(a?: KP, b?: KP) { return (!a || !b) ? NaN : Math.hypot(a.x - b.x, a.y - b.y); }

// Smoothing (heavier)
const last: Record<string, number> = {};
export function smooth(name: string, value: number, alpha = 0.18): number {
  if (!isFinite(value)) return last[name] ?? 0;
  last[name] = last[name] === undefined ? value : alpha * value + (1 - alpha) * last[name];
  return last[name];
}

// Convenience
export function kneeAngle(m: PoseMap)  { return jointAngle(m.hip, m.knee, m.ankle); }
export function elbowAngle(m: PoseMap) { return jointAngle(m.shoulder, m.elbow, m.wrist); }
export function torsoTilt(m: PoseMap)  { return lineTiltFromVertical(m.hip, m.shoulder); }
export function neckAngle(m: PoseMap)  { return lineTiltFromVertical(m.head, m.shoulder); }

// New: stance width (ankles vs hip width), and toe-out angles (each foot)
export function stanceWidthRatio(m: PoseMap) {
  const aw = dist(m.lAnkle, m.rAnkle);
  const hw = dist(m.lHip, m.rHip);
  if (!isFinite(aw) || !isFinite(hw) || hw < 1) return NaN;
  return aw / hw;
}
export function toeOutAngles(m: PoseMap) {
  const footAng = (heel?: KP, toe?: KP) => {
    if (!heel || !toe) return NaN;
    const dy = toe.y - heel.y, dx = toe.x - heel.x;
    return Math.abs(Math.atan2(dx, dy) * 180 / Math.PI); // 0 = forward, bigger = toed-out
  };
  return { left: footAng(m.lHeel, m.lToe), right: footAng(m.rHeel, m.rToe) };
}
