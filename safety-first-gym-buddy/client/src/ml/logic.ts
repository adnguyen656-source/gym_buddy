import {
  indexKeypoints,
  kneeAngle,
  elbowAngle,
  torsoTilt,
  neckAngle,
  pointDeviationFromLine,
  smooth,
  PoseMap,
  KP,
} from "./features";
import { THRESH } from "./constants";
import {
  features as makeFeat,
  classify,
  learn,
  setTrainingMode,
  Label,
} from "./intent";

/* -------------------- Calibration (reverse: relative to YOU) -------------------- */
type Exercise = "squat" | "pushup" | "plank" | "deadbug" | "wallsit";

type Calib = {
  top?:    { hipY?: number; knee?: number; elbow?: number; torso?: number };
  bottom?: { hipY?: number; knee?: number; elbow?: number; torso?: number };
  neutral?:{ knee?: number; elbow?: number; torso?: number };
};
const CAL: Record<Exercise, Calib> = {
  squat: {}, pushup: {}, plank: {}, deadbug: {}, wallsit: {}
};

function snapshot(m: PoseMap) {
  return {
    hipY: m.hip?.y,
    knee: kneeAngle(m),
    elbow: elbowAngle(m),
    torso: torsoTilt(m),
  };
}

let _wantCalib: { ex: Exercise; kind: "top"|"bottom"|"neutral"; until: number } | null = null;
/** UI calls this; we take the first stable snapshot within ~3s */
export function calibStart(ex: Exercise, kind: "top"|"bottom"|"neutral") {
  _wantCalib = { ex, kind, until: performance.now() + 3000 };
}

/* -------------------- Types -------------------- */
export type FrameOut = {
  t: number;
  angles: { knee?: number; elbow?: number; torso?: number };
  signals: { hipHeight?: number; lineDeviation?: number; backContact?: number };
  cue?: string;
};

export type EventOut =
  | { kind: "rep"; index: number; green: boolean; cue?: string }
  | { kind: "sec"; index: number; green: boolean; cue?: string };

type State = {
  started: boolean;
  t0: number;
  hipStart?: number;
  repIndex: number;
  cueHold: { text: string; until: number } | null;
  phase?: "idle" | "down" | "up";
  pPhase?: "idle" | "down" | "up";
  secBox?: { bucket: number; good: number; total: number; secs: number };
  wsSecBox?: { bucket: number; good: number; total: number; secs: number };
  dbPrev?: KP;
  dbDir?: "out" | "in";
  dbLastD?: number;
  prevCentroid?: { x: number; y: number };
  lastMoveT?: number;
  gateAvg: number;
  gated: boolean;

  // no-flicker latch
  goodStreak?: number;
  badStreak?: number;
  latchedOK?: boolean;
};

const baseSt: Omit<State,"started"|"t0"|"repIndex"> = {
  cueHold: null, phase: undefined, pPhase: undefined,
  secBox: undefined, wsSecBox: undefined, dbPrev: undefined,
  dbDir: undefined, dbLastD: undefined, prevCentroid: undefined,
  lastMoveT: undefined, hipStart: undefined,
  gateAvg: 0, gated: false, goodStreak: 0, badStreak: 0, latchedOK: false
};

const state: Record<Exercise, State> = {
  squat:   { started: false, t0: 0, repIndex: 0, ...baseSt },
  pushup:  { started: false, t0: 0, repIndex: 0, ...baseSt },
  plank:   { started: false, t0: 0, repIndex: 0, ...baseSt },
  deadbug: { started: false, t0: 0, repIndex: 0, ...baseSt },
  wallsit: { started: false, t0: 0, repIndex: 0, ...baseSt },
};

function setCue(st: State, text: string) {
  const now = performance.now();
  const hold = THRESH.cueHoldMs ?? 900;
  if (!st.cueHold || st.cueHold.text !== text || now > st.cueHold.until) {
    st.cueHold = { text, until: now + hold };
  }
  return st.cueHold.text;
}

export function enableTraining(on: boolean) { setTrainingMode(on); }

/* -------------------- Main per-frame -------------------- */
export function evaluateFrame(
  pose: { keypoints: KP[] } | null,
  exercise: Exercise
): { frame: FrameOut; event?: EventOut } {
  const st = state[exercise];
  const now = performance.now();
  if (!st.started) { st.started = true; st.t0 = now; }

  let frame: FrameOut = { t: now - st.t0, angles: {}, signals: {} };
  let event: EventOut | undefined;

  if (!pose) return { frame, event };

  const pts = pose.keypoints || [];
  const minScore = THRESH.minKPScore ?? 0.4;
  if (pts.length && pts.every(k => (k.score ?? 0) < minScore)) {
    frame.cue = setCue(st, "Hold still");
    st.gateAvg = 0; st.gated = false;
    return { frame, event };
  }

  // motion estimate
  const cx = pts.reduce((s,k)=>s+(k.x||0),0)/Math.max(1,pts.length);
  const cy = pts.reduce((s,k)=>s+(k.y||0),0)/Math.max(1,pts.length);
  const prev = st.prevCentroid; st.prevCentroid = { x: cx, y: cy };
  const motionPx = prev ? Math.hypot(cx-prev.x, cy-prev.y) : 999;
  const nowT = performance.now();
  if (!st.lastMoveT) st.lastMoveT = nowT;
  if (motionPx > 1.5) st.lastMoveT = nowT;
  const idle = nowT - st.lastMoveT > 1200;

  const m: PoseMap = indexKeypoints(pose);

  // Calibration capture when stable
  if (_wantCalib && _wantCalib.ex === exercise && performance.now() < _wantCalib.until) {
    if (motionPx < 0.8) {
      const snap = snapshot(m);
      if (_wantCalib.kind === "top")     CAL[exercise].top = snap;
      if (_wantCalib.kind === "bottom")  CAL[exercise].bottom = snap;
      if (_wantCalib.kind === "neutral") CAL[exercise].neutral = snap;
      _wantCalib = null;
    }
  }

  // features for your simple classifier (kept)
  const bodyAngleAbs = bodyAngleAbsDeg(m);
  const knee = kneeAngle(m);
  const elbow = elbowAngle(m);
  const torso = torsoTilt(m);
  const torsoAbs = Math.abs(torso);
  const lineDev = lineDeviation(m);
  const stanceFrac = stanceWideEnoughFrac(m);

  const fv = makeFeat(m, motionPx, stanceFrac, {
    bodyAngleAbs, knee, elbow, torsoAbs, lineDev,
  });
  learn(exercise as Label, fv);
  const pred = classify(fv);
  const looksRight = pred.label === exercise && pred.conf > 0.35 && pred.margin > 0.2;

  // gate: keep
  const up = THRESH.gateFlipUp ?? 0.7;
  const down = THRESH.gateFlipDown ?? 0.4;
  st.gateAvg = 0.85 * st.gateAvg + 0.15 * (looksRight ? 1 : 0);
  st.gated = st.gated ? st.gateAvg > down : st.gateAvg > up;

  if (idle || !st.gated) {
    const cue = gateCue(exercise, { bodyAngleAbs, lineDev, stanceFrac });
    frame.cue = setCue(st, cue);
    return { frame, event };
  }

  frame.angles = { knee, elbow, torso };

  if (exercise === "squat") {
    ({ frame, event } = squatLogic(m, st, frame));
  } else if (exercise === "pushup") {
    ({ frame, event } = pushupLogic(m, st, frame));
  } else if (exercise === "plank") {
    ({ frame, event } = plankLogic(m, st, frame));
  } else if (exercise === "deadbug") {
    ({ frame, event } = deadbugLogic(m, st, frame));
  } else if (exercise === "wallsit") {
    ({ frame, event } = wallsitLogic(m, st, frame));
  }

  return { frame, event };
}

/* -------------------- Gate cues -------------------- */
function gateCue(
  ex: Exercise,
  h: { bodyAngleAbs: number; lineDev: number; stanceFrac: number }
) {
  if (ex === "pushup" || ex === "plank") {
    if (h.bodyAngleAbs > 30) return "Lower to plank";
    const lim = THRESH.lineDevMax ?? 8;
    if (h.lineDev > lim) return "Straighten hips and shoulders";
    return ex === "pushup" ? "Hands under shoulders, then lower" : "Hold plank position";
  }
  if (ex === "squat") {
    if (h.bodyAngleAbs < 60) return "Stand tall first";
    if (h.stanceFrac < (THRESH.minStanceFrac ?? 0.25)) return "Widen stance slightly";
    return "Stand tall, then start your descent";
  }
  if (ex === "wallsit") return "Back to wall, slide down";
  if (ex === "deadbug") return "Lie on back, arms and legs up";
  return "Get into position";
}

/* -------------------- Helpers -------------------- */
function bodyAngleAbsDeg(m: PoseMap) {
  if (!m.hip || !m.shoulder) return 90;
  const dy = m.shoulder.y - m.hip.y;
  const dx = m.shoulder.x - m.hip.x;
  return Math.abs((Math.atan2(dy, dx) * 180) / Math.PI);
}
function lineDeviation(m: PoseMap) {
  const s = m.shoulder, a = m.ankle, h = m.hip;
  return pointDeviationFromLine(s, a, h);
}
function stanceWideEnoughFrac(m: PoseMap) {
  const aL = m.ankle?.x ?? 0, aR = m.otherAnkle?.x ?? aL;
  const sL = m.shoulder?.x ?? 0, sR = m.otherShoulder?.x ?? sL;
  const ankleSpan = Math.abs(aL - aR);
  const shoulderSpan = Math.abs(sL - sR) || 1;
  return ankleSpan / shoulderSpan;
}

/* -------------------- Exercise logic (relative) -------------------- */

function squatLogic(m: PoseMap, st: State, frame: FrameOut) {
  const knee = smooth("sq_knee", kneeAngle(m));
  const torso = smooth("sq_torso", torsoTilt(m));
  const hipY = smooth("sq_hipY", m.hip?.y ?? NaN);

  if (st.hipStart === undefined && isFinite(hipY)) st.hipStart = hipY;

  const movedDown =
    st.hipStart !== undefined && hipY > st.hipStart * (1 + (THRESH.repDownFrac ?? 0.10));
  const nearTop =
    st.hipStart !== undefined && hipY <= st.hipStart * (1 + (THRESH.repTopFrac ?? 0.05));

  // Relative depth based on user's own top/bottom calibration
  const topY = CAL.squat.top?.hipY;
  const botY = CAL.squat.bottom?.hipY;
  let depthFrac = 0;
  if (topY !== undefined && botY !== undefined && isFinite(hipY)) {
    const span = Math.max(10, botY - topY);
    depthFrac = (hipY - topY) / span;       // 0 at top, 1 at bottom
  }
  const depthGreen = depthFrac >= 0.65;     // need ~65% of YOUR bottom

  const topTorso = CAL.squat.top?.torso ?? 0;
  const torsoOK = Math.abs(torso - topTorso) <= (THRESH.squatTorsoChangeMax ?? 25);

  // knee cave-in as safety
  const hipX = m.hip?.x ?? 0, kneeX = m.knee?.x ?? 0, ankleX = m.ankle?.x ?? 0;
  const hipAnkleDist = Math.abs(hipX - ankleX) || 1;
  const kneeInside = kneeX < ankleX - (THRESH.kneeCaveFrac ?? 0.08) * hipAnkleDist;
  const kneesOK = !kneeInside;

  const okNow = depthGreen && kneesOK && torsoOK;

  if (st.phase === undefined) st.phase = "idle";
  let showDepth = false;

  if (st.phase === "idle" && movedDown) st.phase = "down";
  if (st.phase === "down") {
    showDepth = true;
    if (!movedDown && depthGreen) st.phase = "up";
  }
  if (st.phase === "up" && nearTop) {
    st.repIndex += 1;
    const green = okNow;
    const ev: EventOut = { kind: "rep", index: st.repIndex, green, cue: green ? undefined : st.cueHold?.text };
    frame.signals.hipHeight = st.hipStart ? hipY - st.hipStart : undefined;
    applyConsecutiveLatch(st, frame, okNow);
    return { frame, event: ev };
  }

  if (showDepth) {
    if (!depthGreen) frame.cue = setCue(st, "Go a little deeper");
    else if (!kneesOK) frame.cue = setCue(st, "Push knees out");
    else if (!torsoOK) frame.cue = setCue(st, "Keep chest up");
    else frame.cue = setCue(st, "");
  } else {
    if (!kneesOK) frame.cue = setCue(st, "Push knees out");
    else frame.cue = setCue(st, "");
  }

  frame.signals.hipHeight = st.hipStart ? hipY - st.hipStart : undefined;
  applyConsecutiveLatch(st, frame, okNow);
  return { frame, event: undefined };
}

function pushupLogic(m: PoseMap, st: State, frame: FrameOut) {
  const elbow = smooth("pu_elbow", elbowAngle(m));
  const lineDev = smooth("pu_line", lineDeviation(m));
  const neck = smooth("pu_neck", neckAngle(m));

  // Relative bottom: your own calibrated bottom elbow + small margin
  const puBotElb = CAL.pushup.bottom?.elbow;
  let depthGreen = false;
  if (puBotElb !== undefined) depthGreen = elbow <= (puBotElb + 10);
  else depthGreen = elbow <= (THRESH.pushupDepthElbowMax ?? 100); // fallback if not calibrated

  const lineOK = lineDev <= (THRESH.lineDevMax ?? 8);
  const neckOK = neck   <= (THRESH.neckMax ?? 18);

  const okNow = depthGreen && lineOK && neckOK;

  if (!depthGreen) frame.cue = setCue(st, "Go a little deeper");
  else if (!lineOK) frame.cue = setCue(st, "Keep a straight line");
  else if (!neckOK) frame.cue = setCue(st, "Tuck chin slightly");
  else frame.cue = setCue(st, "");

  if (st.pPhase === undefined) st.pPhase = "idle";
  if (st.pPhase === "idle" && !depthGreen) st.pPhase = "up";
  if (st.pPhase === "up" && depthGreen) st.pPhase = "down";
  if (st.pPhase === "down" && !depthGreen) {
    st.repIndex += 1;
    const green = lineOK && neckOK;
    const ev: EventOut = { kind: "rep", index: st.repIndex, green, cue: green ? undefined : st.cueHold?.text };
    frame.signals.lineDeviation = lineDev;
    applyConsecutiveLatch(st, frame, okNow);
    return { frame, event: ev };
  }

  frame.signals.lineDeviation = lineDev;
  applyConsecutiveLatch(st, frame, okNow);
  return { frame, event: undefined };
}

function plankLogic(m: PoseMap, st: State, frame: FrameOut) {
  const lineDev = smooth("pl_line", lineDeviation(m));
  const neck = smooth("pl_neck", neckAngle(m));
  const shouldersOverWrists = wristsUnderShoulders(m);

  // Neutral-based guidance optional; primary thresholds still used here
  const lineMax = (THRESH as any).plankLineDevMax ?? (THRESH.lineDevMax ?? 8);
  const neckMax = (THRESH as any).plankNeckMax ?? (THRESH.neckMax ?? 18);

  const ok = lineDev <= lineMax && neck <= neckMax && shouldersOverWrists;
  const okNow = ok;

  const bucket = Math.floor(performance.now() / 1000);
  if (!st.secBox) st.secBox = { bucket, good: 0, total: 0, secs: 0 };
  if (st.secBox.bucket !== bucket) {
    const green =
      st.secBox.total > 0 &&
      st.secBox.good / st.secBox.total >= (THRESH.holdSecondGreenFrac ?? 0.7);
    if (green) {
      st.secBox.secs += 1;
      const ev: EventOut = { kind: "sec", index: st.secBox.secs, green: true };
      st.secBox.bucket = bucket; st.secBox.good = 0; st.secBox.total = 0;

      if (!ok) {
        if (lineDev > lineMax)
          frame.cue = setCue(
            st,
            m.hip && m.shoulder && m.hip.y > m.shoulder.y ? "Lift hips slightly" : "Lower hips slightly"
          );
        else if (!shouldersOverWrists) frame.cue = setCue(st, "Bring shoulders over wrists");
        else if (neck > neckMax) frame.cue = setCue(st, "Relax neck");
        else frame.cue = setCue(st, "");
      } else frame.cue = setCue(st, "");
      applyConsecutiveLatch(st, frame, okNow);
      return { frame, event: ev };
    }
    st.secBox.bucket = bucket; st.secBox.good = 0; st.secBox.total = 0;
  }
  st.secBox.total += 1; if (ok) st.secBox.good += 1;

  frame.signals.lineDeviation = lineDev;
  if (!ok) {
    if (lineDev > lineMax)
      frame.cue = setCue(
        st,
        m.hip && m.shoulder && m.hip.y > m.shoulder.y ? "Lift hips slightly" : "Lower hips slightly"
      );
    else if (!shouldersOverWrists) frame.cue = setCue(st, "Bring shoulders over wrists");
    else if (neck > neckMax) frame.cue = setCue(st, "Relax neck");
  } else frame.cue = setCue(st, "");
  applyConsecutiveLatch(st, frame, okNow);
  return { frame, event: undefined };
}

function deadbugLogic(m: PoseMap, st: State, frame: FrameOut) {
  const back = smooth("db_back", Math.abs((m.hip?.x ?? 0) - (m.shoulder?.x ?? 0)));
  const backOK = back < (THRESH.deadbugBackContactMaxPx ?? 22);

  const p = m.wrist || m.ankle || m.knee;
  const prev = st.dbPrev || p;
  const speed = p ? Math.hypot(p.x - (prev?.x ?? p.x), p.y - (prev?.y ?? p.y)) : 0;
  st.dbPrev = p || prev;
  const slowEnough = speed < (THRESH.deadbugLimbSpeedMaxPx ?? 32);

  const okNow = backOK && slowEnough;

  if (!backOK) frame.cue = setCue(st, "Press lower back into floor");
  else if (!slowEnough) frame.cue = setCue(st, "Slower reach");
  else frame.cue = setCue(st, "");

  const d = p && m.hip ? Math.hypot(p.x - m.hip.x, p.y - m.hip.y) : 0;
  const dSm = smooth("db_wave", d);
  if (!st.dbDir && dSm > (st.dbLastD ?? dSm)) st.dbDir = "out";
  if (st.dbDir === "out" && dSm < (st.dbLastD ?? dSm)) {
    if (backOK && slowEnough) {
      st.repIndex += 1;
      const ev: EventOut = { kind: "rep", index: st.repIndex, green: true };
      st.dbDir = "in"; st.dbLastD = dSm;
      frame.signals.backContact = back;
      applyConsecutiveLatch(st, frame, okNow);
      return { frame, event: ev };
    }
    st.dbDir = "in";
  }
  st.dbLastD = dSm;

  frame.signals.backContact = back;
  applyConsecutiveLatch(st, frame, okNow);
  return { frame, event: undefined };
}

function wallsitLogic(m: PoseMap, st: State, frame: FrameOut) {
  const knee = smooth("ws_knee", kneeAngle(m));
  const shinTilt = smooth("ws_shin", shinAngle(m));
  const backTilt = smooth("ws_back", backVerticality(m));

  // You can also compare to neutral if you take one: CAL.wallsit.neutral?.knee, etc.
  const depthOK = true // depth becomes user-relative if you prefer; keeping safety windows loose
    && (knee >= (THRESH.squatDepthMin ?? 80) && knee <= (THRESH.squatDepthMax ?? 115));
  const shinOK = Math.abs(shinTilt) <= (THRESH.wallsitShinTiltMax ?? 12);
  const backOK = Math.abs(backTilt) <= (THRESH.wallsitBackTiltMax ?? 12);

  const ok = depthOK && shinOK && backOK;
  const okNow = ok;

  if (!depthOK) frame.cue = setCue(st, knee > (THRESH.squatDepthMax ?? 115) ? "Slide down a little" : "Rise slightly");
  else if (!shinOK) frame.cue = setCue(st, "Keep feet under knees");
  else if (!backOK) frame.cue = setCue(st, "Keep back against the wall");
  else frame.cue = setCue(st, "");

  const bucket = Math.floor(performance.now() / 1000);
  if (!st.wsSecBox) st.wsSecBox = { bucket, good: 0, total: 0, secs: 0 };
  if (st.wsSecBox.bucket !== bucket) {
    const green = st.wsSecBox.total > 0 && st.wsSecBox.good / st.wsSecBox.total >= (THRESH.holdSecondGreenFrac ?? 0.7);
    if (green) {
      st.wsSecBox.secs += 1;
      const ev: EventOut = { kind: "sec", index: st.wsSecBox.secs, green: true };
      st.wsSecBox.bucket = bucket; st.wsSecBox.good = 0; st.wsSecBox.total = 0;
      applyConsecutiveLatch(st, frame, okNow);
      return { frame, event: ev };
    }
    st.wsSecBox.bucket = bucket; st.wsSecBox.good = 0; st.wsSecBox.total = 0;
  }
  st.wsSecBox.total += 1; if (ok) st.wsSecBox.good += 1;

  applyConsecutiveLatch(st, frame, okNow);
  return { frame, event: undefined };
}

/* -------------------- Small helpers -------------------- */
function wristsUnderShoulders(m: PoseMap) {
  if (!m.wrist || !m.shoulder) return false;
  const dx = Math.abs(m.wrist.x - m.shoulder.x);
  return dx < (THRESH.wristsUnderShouldersPx ?? 60);
}
function shinAngle(m: PoseMap) {
  if (!m.knee || !m.ankle) return 0;
  const dy = m.knee.y - m.ankle.y;
  const dx = m.knee.x - m.ankle.x;
  return (Math.atan2(dx, dy) * 180) / Math.PI; // zero vertical
}
function backVerticality(m: PoseMap) {
  if (!m.hip || !m.shoulder) return 0;
  const dx = Math.abs(m.hip.x - m.shoulder.x);
  return dx * 0.2;
}

/* -------------------- No-flicker latch (same as before) -------------------- */
function applyConsecutiveLatch(st: State, frame: FrameOut, okNow: boolean) {
  const needGood = THRESH.goodFramesToGreen ?? 12; // ~0.4s @30fps
  const needBad  = THRESH.badFramesToRed ?? 6;     // ~0.2s @30fps

  if (okNow) { st.goodStreak = (st.goodStreak ?? 0) + 1; st.badStreak = 0; }
  else       { st.badStreak  = (st.badStreak  ?? 0) + 1; st.goodStreak = 0; }

  if (!st.latchedOK && (st.goodStreak ?? 0) >= needGood) st.latchedOK = true;
  if (st.latchedOK && (st.badStreak ?? 0) >= needBad)   st.latchedOK = false;

  if (st.latchedOK) frame.cue = "";
}
