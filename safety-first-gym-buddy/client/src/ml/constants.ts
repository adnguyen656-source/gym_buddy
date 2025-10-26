// client/src/ml/constants.ts
export const THRESH = {
  // generic
  minKPScore: 0.40,
  cueHoldMs: 800,
  holdSecondGreenFrac: 0.70,

  // squat (tuned so your demo “just works”)
  // bottom knee roughly 80–115°, descent about 12% hip travel
  squatDepthMin: 80,
  squatDepthMax: 115,
  repDownFrac: 0.12,     // moved down ≥12% from start
  repTopFrac: 0.06,      // returned within 6% of start
  kneeCaveFrac: 0.05,    // knee inside ankle by >5% of hip–ankle span = not ok
  squatTorsoChangeMax: 20, // torso tilt change limit (deg)

  // push-up (lenient so it doesn’t nag)
  pushupDepthElbowMax: 105, // count depth when elbow ≤105°
  lineDevMax: 10,           // hip off shoulder–ankle line (deg)
  neckMax: 18,              // neck flex (deg)

  // plank (same alignment limits as push-up)
  // we reuse lineDevMax and neckMax above

  // wall sit (looser verticality)
  wallsitShinTiltMax: 12,   // shins within 12° of vertical
  wallsitBackTiltMax: 12,   // back within 12° of vertical

  // dead bug (lenient floor contact + slower is better)
  deadbugBackContactMaxPx: 20,
  deadbugLimbSpeedMaxPx: 28,
};
