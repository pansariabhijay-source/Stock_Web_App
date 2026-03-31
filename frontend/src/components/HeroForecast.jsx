import React from 'react';
import { motion } from 'framer-motion';
import { Activity, ShieldCheck, Zap } from 'lucide-react';
import { cn } from '../lib/utils';

const HeroForecast = ({ regime, regimeSince, volatility }) => {
  const isBull = regime?.toLowerCase() === 'bull';
  const isBear = regime?.toLowerCase() === 'bear';
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="col-span-1 lg:col-span-4 relative overflow-hidden rounded-2xl bg-white/5 border border-white/10 p-8 backdrop-blur-xl"
    >
      {/* Background Glow */}
      <div className={cn(
        "absolute -top-32 -left-32 w-64 h-64 rounded-full mix-blend-screen filter blur-[100px] opacity-30",
        isBull ? "bg-emerald-500" : isBear ? "bg-rose-500" : "bg-blue-500"
      )} />

      <h2 className="text-slate-400 font-label tracking-widest text-xs uppercase mb-8">
        Market Core State
      </h2>

      <div className="flex items-end justify-between mb-8">
        <div>
          <div className="text-4xl font-headline font-extrabold text-white capitalize tracking-tight mb-2">
            {regime} Market
          </div>
          <div className="text-slate-500 text-sm font-body">
            Detected since: {regimeSince || "Recently"}
          </div>
        </div>
        
        <div className={cn(
          "w-12 h-12 rounded-full flex items-center justify-center border",
          isBull ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400" : 
          isBear ? "bg-rose-500/20 border-rose-500/50 text-rose-400" : 
          "bg-blue-500/20 border-blue-500/50 text-blue-400"
        )}>
          {isBull ? <TrendingUpIcon /> : isBear ? <TrendingDownIcon /> : <Activity />}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-black/20 rounded-xl p-4 border border-white/5">
          <div className="flex items-center gap-2 text-slate-400 font-label text-xs mb-2">
            <ShieldCheck size={14} /> System Confidence
          </div>
          <div className="text-2xl font-body font-semibold text-slate-200">92.4%</div>
        </div>
        <div className="bg-black/20 rounded-xl p-4 border border-white/5">
          <div className="flex items-center gap-2 text-slate-400 font-label text-xs mb-2">
            <Zap size={14} /> Volatility Index
          </div>
          <div className="text-2xl font-body font-semibold text-slate-200">{volatility || "Normal"}</div>
        </div>
      </div>
    </motion.div>
  );
};

// Quick inline icons if lucide doesn't have exactly what we want imported above
const TrendingUpIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg>
);
const TrendingDownIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"></polyline><polyline points="16 17 22 17 22 11"></polyline></svg>
);

export default HeroForecast;
