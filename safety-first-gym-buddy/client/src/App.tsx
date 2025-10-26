import React, { useEffect, useRef, useState } from "react";
import { estimatePoses } from "./ml/poseLoader";
import { evaluateFrame, enableTraining } from "./ml/logic";

type Exercise = "squat" | "pushup" | "plank" | "deadbug" | "wallsit";

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [exercise, setExercise] = useState<Exercise>("squat");
  const [useDemo, setUseDemo] = useState(true);
  const [count, setCount] = useState(0);
  const [cue, setCue] = useState("");
  const [calibUntil, setCalibUntil] = useState(0);
  const [ok, setOk] = useState(false);
  const [mute, setMute] = useState(true); // default muted to avoid browser blocks
  const lastCueRef = useRef("");

  // load webcam or demo and size canvas
  useEffect(() => {
    enableTraining(useDemo); // learn centroids when demo plays

    const v = videoRef.current;
    if (!v) return;

    const fit = () => {
      const c = canvasRef.current;
      if (!c) return;
      c.width = v.videoWidth || 640;
      c.height = v.videoHeight || 480;
    };

    if (useDemo) {
      const demoMap: Record<Exercise, string> = {
        squat: "/demo/squat_good_side.mp4",
        pushup: "/demo/pushup_good_side.mp4",
        plank: "/demo/plank_good_side.mp4",
        deadbug: "/demo/deadbug_good_side.mp4",
        wallsit: "/demo/wallsit_good_side.mp4",
      };
      v.srcObject = null;
      v.src = demoMap[exercise];
      v.loop = true;
      v.muted = true;
      v.onloadedmetadata = () => {
        fit();
        v.play().catch(() => {});
      };
      v.load();
    } else {
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: false })
        .then((s) => {
          v.src = "";
          v.srcObject = s;
          v.onloadedmetadata = () => {
            fit();
            v.play().catch(() => {});
          };
        })
        .catch(console.error);
    }

    v.onresize = fit;
  }, [useDemo, exercise]);

  // main loop
  useEffect(() => {
    let raf = 0;
    const loop = async () => {
      const v = videoRef.current;
      if (v) {
        const pose = await estimatePoses(v);
        const { frame, event } = evaluateFrame(pose as any, exercise);
        drawOverlay(pose, canvasRef.current, frame);

        const goodNow = !(frame.cue && frame.cue.length > 0);
        setOk(goodNow);

        if (event) setCount(event.index);

        if (performance.now() > calibUntil) setCue(frame.cue || "");
        else setCue("Calibrating…");

        // optional speech every four seconds based on current cue
        // keeps things simple and light
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [exercise, calibUntil]);

  return (
    <div style={{ padding: 16 }}>
      <div style={{ marginBottom: 8 }}>
        <select
          value={exercise}
          onChange={(e) => {
            const ex = e.target.value as Exercise;
            setExercise(ex);
            setCount(0);
            setCue("Calibrating…");
            setCalibUntil(performance.now() + 2000);
            lastCueRef.current = "Calibrating…";
          }}
        >
          <option value="squat">Squat</option>
          <option value="pushup">Push-up</option>
          <option value="plank">Plank</option>
          <option value="deadbug">Dead bug</option>
          <option value="wallsit">Wall sit</option>
        </select>
        <button onClick={() => setUseDemo((s) => !s)} style={{ marginLeft: 8 }}>
          {useDemo ? "Use webcam" : "Use demo video"}
        </button>
        <button onClick={() => setMute((m) => !m)} style={{ marginLeft: 8 }}>
          {mute ? "Voice off" : "Voice on"}
        </button>
      </div>

      <div
        style={{
          position: "relative",
          width: 640,
          height: 480,
          border: `4px solid ${ok ? "green" : "red"}`,
        }}
      >
        <video
          ref={videoRef}
          width={640}
          height={480}
          style={{ position: "absolute", left: 0, top: 0 }}
          playsInline
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{ position: "absolute", left: 0, top: 0 }}
        />
      </div>

      <div style={{ marginTop: 12, fontSize: 18 }}>
        Count: <b>{count}</b> | Tip: {cue || "Follow the on screen guidance"}
      </div>
    </div>
  );
}

function drawOverlay(pose: any, canvas: HTMLCanvasElement | null, frame: any) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (pose?.keypoints) {
    ctx.lineWidth = 2;
    (pose.keypoints || []).forEach((k: any) => {
      if ((k.score ?? 0) > 0.3) {
        ctx.beginPath();
        ctx.arc(k.x, k.y, 3, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
  } else {
    ctx.font = "14px sans-serif";
    ctx.fillStyle = "black";
    ctx.fillText("no pose yet…", 8, 18);
  }

  const a = frame?.angles || {};
  const s = frame?.signals || {};
  const lines = [
    `knee: ${a.knee?.toFixed?.(1) ?? "-"}`,
    `elbow: ${a.elbow?.toFixed?.(1) ?? "-"}`,
    `torso: ${a.torso?.toFixed?.(1) ?? "-"}`,
    `lineDev: ${s.lineDeviation?.toFixed?.(1) ?? "-"}`,
    `hipHeight: ${s.hipHeight?.toFixed?.(1) ?? "-"}`,
    `backContact: ${s.backContact?.toFixed?.(1) ?? "-"}`,
  ];
  ctx.font = "14px sans-serif";
  ctx.fillStyle = "black";
  lines.forEach((txt, i) => ctx.fillText(txt, 8, 18 + i * 16));
}
