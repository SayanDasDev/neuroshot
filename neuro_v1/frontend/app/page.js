'use client';
import { useState, useEffect, useRef } from 'react';

export default function Home() {
  // --- Game Config & Constants ---
  const WIDTH = 800;
  const HEIGHT = 400;
  const GROUND_Y = 350;
  const G = 9.8;
  const DT = 0.03; // Time step matches Python Env

  // --- State ---
  const canvasRef = useRef(null);
  
  // Physics State
  const [force, setForce] = useState(70);
  const [angle, setAngle] = useState(45);
  const [wind, setWind] = useState(0);
  
  // Basket State
  const [basket, setBasket] = useState({ x: 600, speed: -8, amp: 30, freq: 1, base: 600 });
  
  // Ball State
  // We track 't' (simulation time) to keep basket and ball in sync
  const [simState, setSimState] = useState({ 
    x: 50, y: GROUND_Y, 
    vx: 0, vy: 0, 
    active: false, 
    t: 0, // Simulation time accumulator
    path: [] 
  });
  
  // Meta State
  const [status, setStatus] = useState("Ready");
  const [aiThinking, setAiThinking] = useState(false);
  const [score, setScore] = useState(0);

  // --- Initialization ---
  useEffect(() => {
    resetLevel();
  }, []);

  const resetLevel = () => {
    const newWind = (Math.random() * 6 - 3); // -3 to 3
    const newBase = 400 + Math.random() * 300; // 400 to 700
    setWind(newWind);
    setBasket({
      x: newBase,
      speed: -10 + Math.random() * 5, // -10 to -5
      amp: 20 + Math.random() * 30,   // 20 to 50
      freq: 0.5 + Math.random() * 1.5, // 0.5 to 2.0
      base: newBase
    });
    resetBall();
  };

  const resetBall = () => {
    setSimState({ 
        x: 50, y: GROUND_Y, 
        vx: 0, vy: 0, 
        active: false, 
        t: 0, 
        path: [] 
    });
    setStatus("Ready");
  };

  // --- Physics Loop ---
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    // Mutable ref to track physics between renders without triggering re-renders
    // This allows smooth 60FPS animation independent of React state updates
    let physics = { ...simState }; 
    // We only update the local 'physics' object during the loop, 
    // and sync back to React state only when the shot ends.

    const render = () => {
      // Clear
      ctx.fillStyle = '#0f172a'; // Slate 900
      ctx.fillRect(0, 0, WIDTH, HEIGHT);

      // Draw Ground
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, GROUND_Y);
      ctx.lineTo(WIDTH, GROUND_Y);
      ctx.stroke();

      // Draw Wind Gauge
      const cx = 400, cy = 50;
      ctx.strokeStyle = '#475569';
      ctx.beginPath();
      ctx.moveTo(cx - 80, cy);
      ctx.lineTo(cx + 80, cy);
      ctx.stroke();
      if (Math.abs(wind) > 0.1) {
        ctx.strokeStyle = wind > 0 ? '#22c55e' : '#ef4444'; // Green right, Red left
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + (wind * 20), cy);
        ctx.stroke();
      }

      // --- BASKET PHYSICS ---
      let currentBasketX;

      if (physics.active) {
        // MOVING: If shot is active, update time and calculate position
        physics.t += DT; 
        
        // x(t) = base + (v * t) + (A * sin(f * t))
        const linearMove = basket.speed * physics.t;
        const oscillation = basket.amp * Math.sin(basket.freq * physics.t);
        currentBasketX = basket.base + linearMove + oscillation;
      } else {
        // IDLE: Show the "Start" position (t=0)
        // This ensures the user aims at the same setup the AI sees.
        currentBasketX = basket.base; 
      }

      // Draw Basket
      ctx.fillStyle = '#06b6d4'; // Cyan
      ctx.fillRect(currentBasketX, GROUND_Y - 10, 60, 10);
      // Center line
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(currentBasketX + 30, GROUND_Y - 10);
      ctx.lineTo(currentBasketX + 30, GROUND_Y);
      ctx.stroke();


      // --- BALL PHYSICS ---
      if (physics.active) {
        // x += vx * dt + 0.5 * wind * dt^2
        let newX = physics.x + (physics.vx * DT) + (0.5 * wind * DT * DT);
        // y -= vy * dt - 0.5 * g * dt^2
        let newY = physics.y - ((physics.vy * DT) - (0.5 * G * DT * DT));
        
        // Update velocity (for next frame, though Euler doesn't strictly need it for Pos)
        physics.vx = physics.vx + wind * DT;
        physics.vy = physics.vy - G * DT;

        // Check Ground Collision
        if (newY >= GROUND_Y || newX > WIDTH) {
          newY = GROUND_Y;
          physics.active = false; // Stop
          
          // Check Hit (Distance from Ball to Basket Center)
          const basketCenter = currentBasketX + 30;
          const dist = Math.abs(newX - basketCenter);
          
          if (dist < 25) { // Threshold matches Python Env
            setStatus("HIT! (+100)");
            setScore(s => s + 1);
          } else {
            setStatus(`MISS (Err: ${dist.toFixed(1)})`);
          }
          // Sync final state back to React
          setSimState({...physics}); 
        }

        physics.x = newX;
        physics.y = newY;
        physics.path.push({x: newX, y: newY});
      }

      // Draw Path
      if (physics.path.length > 0) {
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.5)'; // Pink
        ctx.lineWidth = 2;
        ctx.beginPath();
        physics.path.forEach((p, i) => {
            if(i===0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
      }

      // Draw Ball
      ctx.fillStyle = '#f97316'; // Orange
      ctx.beginPath();
      ctx.arc(physics.x, physics.y, 6, 0, Math.PI * 2);
      ctx.fill();

      // If idle, draw the aiming line
      if (!physics.active) {
         const rad = angle * (Math.PI / 180);
         const barrelX = 50 + 40 * Math.cos(rad);
         const barrelY = GROUND_Y - 40 * Math.sin(rad);
         ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
         ctx.lineWidth = 2;
         ctx.beginPath();
         ctx.moveTo(50, GROUND_Y);
         ctx.lineTo(barrelX, barrelY);
         ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationFrameId);
  }, [simState.active, basket, wind, force, angle]); // Dependency array

  const handleManualShoot = () => {
    // Convert inputs to velocity
    const theta = angle * (Math.PI / 180);
    const vx = force * Math.cos(theta);
    const vy = force * Math.sin(theta);
    
    // Reset simulation time to 0 and activate
    setSimState({ 
        x: 50, y: GROUND_Y, 
        vx, vy, 
        active: true, 
        t: 0, 
        path: [] 
    });
    setStatus("Flying...");
  };

  const handleAIShot = async () => {
    setAiThinking(true);
    setStatus("AI Thinking...");
    
    // Normalize Observation matches Python Env exactly
    const obs = [
      basket.base / WIDTH,         // basket_x
      basket.speed / 20.0,         // basket_speed
      wind / 5.0,                  // wind_force
      basket.amp / 50.0,           // amplitude
      basket.freq / 2.0,           // frequency
      50.0 / WIDTH,                // ball_x
      GROUND_Y / HEIGHT            // ball_y
    ];

    try {
      // Ensure your app.py is running on port 8000
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ data: obs })
      });
      
      const data = await res.json();
      const action = data.action; 
      
      // Decode Action
      const predForce = ((action[0] + 1) / 2) * 80 + 30;
      const predAngle = ((action[1] + 1) / 2) * 70 + 15;
      
      setForce(predForce);
      setAngle(predAngle);
      
      setAiThinking(false);
      
      // Small delay so user sees the settings change before firing
      setTimeout(() => {
          handleManualShoot();
      }, 500);
      
    } catch (e) {
      console.error(e);
      setStatus("API Error (Is app.py running?)");
      setAiThinking(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-12 bg-gray-950 text-white font-sans">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl lg:static lg:w-auto lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-slate-800/30">
          <code className="font-bold">NeuroShot Web Interface</code>
        </p>
        <div className="flex place-items-center gap-2">
            <span className="text-xl font-bold text-emerald-400">Score: {score}</span>
        </div>
      </div>

      <div className="relative flex place-items-center mt-5">
        <canvas 
            ref={canvasRef} 
            width={WIDTH} 
            height={HEIGHT}
            className="border-2 border-cyan-500 rounded-lg shadow-[0_0_20px_rgba(6,182,212,0.5)] bg-slate-900"
        />
      </div>

      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-4 lg:text-left mt-10 gap-5">
        
        {/* Controls Card */}
        <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30 col-span-2">
          <h2 className="mb-3 text-2xl font-semibold">Controls</h2>
          
          <div className="space-y-4">
            <div>
                <label className="block text-sm font-medium text-slate-400">Force: {force.toFixed(1)}</label>
                <input 
                    type="range" min="30" max="110" step="0.5" 
                    value={force} onChange={e => setForce(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
            </div>
            <div>
                <label className="block text-sm font-medium text-slate-400">Angle: {angle.toFixed(1)}°</label>
                <input 
                    type="range" min="15" max="85" step="0.5" 
                    value={angle} onChange={e => setAngle(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                />
            </div>
          </div>
          
          <div className="flex gap-4 mt-6">
            <button 
                onClick={handleManualShoot}
                disabled={simState.active}
                className="px-6 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded font-bold transition-all"
            >
                Shoot
            </button>
            <button 
                onClick={resetLevel}
                className="px-6 py-2 bg-red-900/50 hover:bg-red-900 rounded font-bold text-red-200 transition-all"
            >
                Reset Level
            </button>
          </div>
        </div>

        {/* AI Card */}
        <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30 col-span-2 border-l-4 border-l-emerald-500 bg-neutral-900/50">
          <h2 className="mb-3 text-2xl font-semibold text-emerald-400">AI Pilot</h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50 mb-4">
            Ask the PPO Agent for the optimal trajectory.
          </p>
          
          <div className="flex flex-col gap-4">
            <div className="text-sm font-mono text-gray-400">
                Status: <span className="text-white font-bold">{status}</span>
            </div>
            
            <button 
                onClick={handleAIShot}
                disabled={aiThinking || simState.active}
                className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-bold text-lg shadow-lg shadow-emerald-900/20 transition-all flex items-center justify-center gap-2"
            >
                {aiThinking ? "Thinking..." : "✨ AI SHOT ✨"}
            </button>
          </div>
        </div>

      </div>
    </main>
  );
}