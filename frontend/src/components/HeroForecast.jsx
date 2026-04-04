import React from 'react';
import { Activity, ShieldCheck, Zap, TrendingUp, TrendingDown } from 'lucide-react';

const HeroForecast = ({ regime, regimeSince, volatility }) => {
  const isBull = regime?.toLowerCase() === 'bull';
  const isBear = regime?.toLowerCase() === 'bear';
  
  return (
    <div className="col-span-1 xl:col-span-4 rounded-2xl bg-surface-container p-6 animate-fade-in">
      
      <p className="text-[10px] font-semibold uppercase tracking-widest text-on-surface-variant/50 mb-6">
        Market Regime
      </p>

      <div className="flex items-end justify-between mb-6">
        <div>
          <div className="text-3xl font-bold text-on-surface capitalize tracking-tight mb-1">
            {regime || '...'} Market
          </div>
          <div className="text-on-surface-variant/60 text-sm">
            Active since {regimeSince || "—"}
          </div>
        </div>
        
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
          isBull ? "bg-gain/10 text-gain" : 
          isBear ? "bg-loss/10 text-loss" : 
          "bg-primary/10 text-primary"
        }`}>
          {isBull ? <TrendingUp size={20} /> : isBear ? <TrendingDown size={20} /> : <Activity size={20} />}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface-container-high rounded-lg p-4">
          <div className="flex items-center gap-2 text-on-surface-variant/60 text-[11px] font-medium mb-2">
            <ShieldCheck size={13} /> Confidence
          </div>
          <div className="text-xl font-semibold text-on-surface tabular-nums">92.4%</div>
        </div>
        <div className="bg-surface-container-high rounded-lg p-4">
          <div className="flex items-center gap-2 text-on-surface-variant/60 text-[11px] font-medium mb-2">
            <Zap size={13} /> Volatility
          </div>
          <div className="text-xl font-semibold text-on-surface">{volatility || "Normal"}</div>
        </div>
      </div>
    </div>
  );
};

export default HeroForecast;
