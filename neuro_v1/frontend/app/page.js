'use client';
import { useState, useEffect, useRef } from 'react';

export default function Home() {
  // --- Game Config & Constants ---
  const WIDTH = 800;
  const HEIGHT = 400;
  const GROUND_Y = 350;
  const G = 9.8;
  const DT = 0.03; 

  // --- State ---
  const canvasRef = useRef(null);
  
  // Physics State
  const [force, setForce] = useState(70);
  const [angle, setAngle] = useState(45);
  const [wind, setWind] = useState(0);
  
  // Basket State
  const [basket, setBasket] = useState({ x: 600, speed: -8, amp: 30, freq: 1, base: 600 });
  
  // Ball State
  const [ball, setBall] = useState({ x: 50, y: GROUND_Y, vx: 0, vy: 0, active: false, path: [] });
  
  // Meta State
  const [status, setStatus] = useState("Ready");
  const [aiThinking, setAiThinking] = useState(false);
  const [score, setScore] = useState(0);

  // --- Initialization ---
  useEffect(() => {
    resetLevel();
  }, []);

  const resetLevel = () => {
    // Randomize Environment similar to Env reset
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
    setBall({ x: 50, y: GROUND_Y, vx: 0, vy: 0, active: false, path: [] });
    setStatus("Ready");
  };

  // --- Physics Loop ---
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let t = 0; // Local time for basket oscillation

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
        ctx.strokeStyle = '#22c55e'; // Green
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + (wind * 20), cy);
        ctx.stroke();
      }

      // Update Basket Position (Oscillation)
      // Note: React state update is async, so we use local math mostly or assume rough sync
      // Ideally we track 'global time' state but for visual sim we increment strictly
      t += DT;
      
      // Calculate current basket position based on initial params (simplified for loop)
      // Real Env uses continuous time. Here we just animate 'live'
      const oscillation = basket.amp * Math.sin(basket.freq * (Date.now() / 1000)); 
      const currentBasketX = basket.base + oscillation; // Simplified movement for visual aid

      // Draw Basket
      ctx.fillStyle = '#06b6d4'; // Cyan
      ctx.fillRect(currentBasketX, GROUND_Y - 10, 60, 10);

      // Update Ball Physics if active
      if (ball.active) {
        // Simple Euler integration
        // x += vx * dt + 0.5 * wind * dt^2 (Env logic)
        // Environment uses absolute time t. We stick to iterative for simple animation
        
        let newX = ball.x + (ball.vx * DT) + (0.5 * wind * DT * DT);
        let newY = ball.y - ((ball.vy * DT) - (0.5 * G * DT * DT));
        
        // Update Velocity (approx)
        let newVx = ball.vx + wind * DT;
        let newVy = ball.vy - G * DT;

        // Check Collision / Ground
        if (newY >= GROUND_Y) {
          newY = GROUND_Y;
          ball.active = false; // Stop
          
          // Check Hit
          const dist = Math.abs(newX - currentBasketX);
          if (dist < 30) {
            setStatus("HIT! (+100)");
            setScore(s => s + 1);
          } else {
            setStatus("MISS");
          }
        }

        // Mutation (safe inside render loop for performance, sync to state occasionally?)
        // Actually modifying state in rAF is bad. We use local vars and ref.
        ball.x = newX;
        ball.y = newY;
        ball.vx = newVx;
        ball.vy = newVy;
        ball.path.push({x: newX, y: newY});
      }

      // Draw Path
      if (ball.path.length > 0) {
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.5)'; // Pink
        ctx.lineWidth = 2;
        ctx.beginPath();
        ball.path.forEach((p, i) => {
            if(i===0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
      }

      // Draw Ball
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, 6, 0, Math.PI * 2);
      ctx.fill();

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationFrameId);
  }, [ball, basket, wind, force, angle]); // Re-bind if config changes

  const handleManualShoot = () => {
    resetBall();
    // Convert Force/Angle to Vx/Vy
    // Env Logic: 
    // v0 = force
    // theta = radians(angle)
    const theta = angle * (Math.PI / 180);
    const vx = force * Math.cos(theta);
    const vy = force * Math.sin(theta);
    
    // Trigger loop
    // slightly hacky: wait 1 frame for reset? 
    setTimeout(() => {
        setBall({ x: 50, y: GROUND_Y, vx, vy, active: true, path: [] });
        setStatus("Flying...");
    }, 50);
  };

  const handleAIShot = async () => {
    setAiThinking(true);
    setStatus("AI Thinking...");
    
    // 1. Normalize Observation
    // [basket_x, basket_speed, wind, amp, freq, ball_x, ball_y]
    // Note: Env uses Date-based or strict time-based basket pos. 
    // For AI to work, we must send current basket state.
    
    // Normalization factors from env code
    const obs = [
      basket.base / WIDTH,           // basket_x (approx base)
      basket.speed / 20.0,           // speed
      wind / 5.0,                    // wind
      basket.amp / 50.0,             // amp
      basket.freq / 2.0,             // freq
      50.0 / WIDTH,                  // ball_x fixed start
      GROUND_Y / HEIGHT              // ball_y fixed start
    ];

    try {
      const res = await fetch('http://127.0.0.1:8001/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ data: obs })
      });
      
      const data = await res.json();
      const action = data.action; // [force_norm, angle_norm]
      
      // 2. Decode Action
      // v0 = ((action[0] + 1) / 2) * 80 + 30
      // theta = ((action[1] + 1) / 2) * 70 + 15
      
      const predForce = ((action[0] + 1) / 2) * 80 + 30;
      const predAngle = ((action[1] + 1) / 2) * 70 + 15;
      
      setForce(predForce);
      setAngle(predAngle);
      
      setAiThinking(false);
      handleManualShoot(); // Auto-fire
      
    } catch (e) {
      console.error(e);
      setStatus("API Error");
      setAiThinking(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-12 bg-gray-950 text-white font-sans">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl lg:static lg:w-auto lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-slate-800/30">
          <code className="font-bold">NeuroShot End-to-End V1</code>
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
                className="px-6 py-2 bg-slate-700 hover:bg-slate-600 rounded font-bold transition-all"
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
                disabled={aiThinking}
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
