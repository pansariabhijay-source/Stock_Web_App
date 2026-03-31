import React from 'react';

const Analysis = () => {
  return (
    <div className="px-8 py-8 space-y-8">
      {/* Stock Header Section */}
      <section className="flex flex-col lg:flex-row justify-between items-start lg:items-end gap-6 pb-4 border-b border-outline-variant/15">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <span className="px-2 py-1 bg-surface-container-highest rounded text-[10px] font-bold text-primary tracking-wider uppercase">NSE: RELIANCE</span>
            <h2 className="text-3xl font-headline font-extrabold text-white tracking-tight">Reliance Industries Ltd.</h2>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-baseline gap-2">
              <span className="text-4xl font-black tabular-nums text-white">2,945.20</span>
              <span className="text-sm font-medium text-slate-400 uppercase">INR</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/10">
              <span className="material-symbols-outlined text-secondary text-sm" style={{fontVariationSettings: "'FILL' 1"}}>trending_up</span>
              <span className="text-secondary font-bold tabular-nums">+42.15 (1.45%)</span>
              <span className="text-[10px] text-on-secondary-fixed-variant font-medium ml-1">Today</span>
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          <button className="px-6 py-2.5 bg-gradient-to-br from-primary to-primary-container text-on-primary font-bold rounded-lg text-sm hover:opacity-90 transition-opacity">Add to Portfolio</button>
          <button className="px-4 py-2.5 bg-surface-container text-on-surface font-bold rounded-lg text-sm border border-outline-variant/30 hover:bg-surface-bright transition-colors">Alerts</button>
        </div>
      </section>

      {/* Main Intelligence Layout */}
      <div className="grid grid-cols-12 gap-8">
        {/* Left Column: Chart & Backtest */}
        <div className="col-span-12 lg:col-span-8 space-y-8">
          {/* Chart Module */}
          <div className="bg-surface-container rounded-2xl p-6 relative overflow-hidden">
            <div className="flex justify-between items-center mb-6">
              <div className="flex gap-4">
                <button className="text-xs font-bold text-primary border-b-2 border-primary pb-1">Price Analysis</button>
                <button className="text-xs font-bold text-slate-500 hover:text-slate-300">Volume Flow</button>
                <button className="text-xs font-bold text-slate-500 hover:text-slate-300">Option Chain</button>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-secondary"></span>
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Live Terminal</span>
              </div>
            </div>

            {/* Candlestick Visualization Substitute */}
            <div className="h-80 w-full relative group">
              <img className="w-full h-full object-cover rounded-xl opacity-40 mix-blend-screen" alt="candlestick chart" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDp4a4KG1hSTSXUAp8ews4WZb0b5OKThBsesJOhpadbPPAnItTriOtUBoJ_LX520hdWtAiEPalzMVGIzCmk-4TbruDVSVwemWptisWgygQja97uoff3u8LHXR1pdsXiS27d_xSYKWnPQm-s8B2_l6zwC8jninkd5uI3AX7ok8-xaLHYnOg31XsyGmyEN9W0JGVtqk5mkTbBiq3cpLLv6g-_dGtp2F_n2kDaenapV-TgzSPZGS240FlENIiBCfIH12DM5fbBk0WuLsQc" />
              
              {/* Institutional Zone Overlays (Visual elements) */}
              <div className="absolute top-10 left-1/4 right-1/4 h-12 bg-secondary/5 border-y border-dashed border-secondary/20 flex items-center justify-center">
                <span className="text-[10px] text-secondary font-bold uppercase tracking-tighter">Institutional Accumulation Zone</span>
              </div>
              
              <div className="absolute bottom-16 left-10 right-1/2 h-8 bg-error/5 border-y border-dashed border-error/20 flex items-center justify-center">
                <span className="text-[10px] text-error font-bold uppercase tracking-tighter">Liquidating Supply Zone</span>
              </div>

              {/* Floating Tooltip Simulation */}
              <div className="absolute top-1/2 left-2/3 backdrop-blur-md bg-surface-variant/60 p-3 rounded-xl border border-white/10 shadow-2xl z-10 pointer-events-none">
                <div className="text-[10px] font-bold text-primary mb-1">ST-CORE MODEL</div>
                <div className="text-white font-bold text-sm">Forecast: Bullish Extension</div>
                <div className="text-[10px] text-slate-400">Confidence Score: 89.4%</div>
              </div>
            </div>
          </div>

          {/* Backtest Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-surface-container-low rounded-2xl p-5 border border-outline-variant/10">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Model Accuracy</p>
              <div className="flex items-baseline gap-2">
                <h4 className="text-2xl font-black text-white tabular-nums">92.4%</h4>
                <span className="text-[10px] text-secondary font-bold">+1.2% v/s Avg</span>
              </div>
              <p className="text-xs text-slate-400 mt-3">Accuracy based on 450 historic trades over 24 months.</p>
            </div>
            <div className="bg-surface-container-low rounded-2xl p-5 border border-outline-variant/10">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Historical Alpha</p>
              <div className="flex items-baseline gap-2">
                <h4 className="text-2xl font-black text-white tabular-nums">+18.2%</h4>
                <span className="text-[10px] text-secondary font-bold">Annualized</span>
              </div>
              <p className="text-xs text-slate-400 mt-3">Returns generated above the sector benchmark index.</p>
            </div>
            <div className="bg-surface-container-low rounded-2xl p-5 border border-outline-variant/10">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Risk Profile</p>
              <div className="flex items-baseline gap-2">
                <h4 className="text-2xl font-black text-white uppercase">Moderate</h4>
              </div>
              <div className="w-full h-1.5 bg-surface-container-highest rounded-full mt-4 overflow-hidden">
                <div className="h-full w-2/3 bg-primary rounded-full"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Intelligence Hub & Targets */}
        <div className="col-span-12 lg:col-span-4 space-y-8">
          {/* Intelligence Hub */}
          <div className="bg-surface-container rounded-2xl p-6 h-fit">
            <div className="flex items-center gap-3 mb-6">
              <span className="material-symbols-outlined text-primary">psychology</span>
              <h3 className="text-lg font-headline font-bold text-white">The Intelligence Hub</h3>
            </div>
            
            <div className="space-y-6">
              <div className="relative pl-6 border-l-2 border-primary/20">
                <div className="absolute -left-1.5 top-0 w-3 h-3 rounded-full bg-primary border-4 border-background"></div>
                <h5 className="text-sm font-bold text-white mb-1">Sector Strength</h5>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Energy benchmarks are trading at a 15% premium to their 50-day moving average. Reliance is leading this decoupling, showing institutional appetite for large-cap stability.
                </p>
              </div>
              
              <div className="relative pl-6 border-l-2 border-secondary/20">
                <div className="absolute -left-1.5 top-0 w-3 h-3 rounded-full bg-secondary border-4 border-background"></div>
                <h5 className="text-sm font-bold text-white mb-1">Institutional Flow</h5>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Large block trades have concentrated around the 2,920 support level. This suggests 'smart money' is defending current valuations against short-term volatility.
                </p>
              </div>
              
              <div className="relative pl-6 border-l-2 border-on-surface-variant/20">
                <div className="absolute -left-1.5 top-0 w-3 h-3 rounded-full bg-on-surface-variant border-4 border-background"></div>
                <h5 className="text-sm font-bold text-white mb-1">Price Momentum</h5>
                <p className="text-xs text-slate-400 leading-relaxed">
                  The stock has cleared its immediate technical hurdle. We observe a rare convergence of volume and price expansion, typical of an early-stage trend breakout.
                </p>
              </div>
            </div>
            
            <div className="mt-8 p-4 bg-primary/5 rounded-xl border border-primary/10">
              <div className="flex items-center gap-2 mb-2">
                <span className="material-symbols-outlined text-primary text-sm">verified</span>
                <span className="text-xs font-bold text-primary">Core Recommendation</span>
              </div>
              <p className="text-sm font-semibold text-white">Accumulate on minor pullbacks. The structure remains intact for a medium-term climb.</p>
            </div>
          </div>

          {/* Forecast Targets */}
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest ml-2">Forecast Targets</h4>
            
            {/* 1D Target */}
            <div className="bg-surface-container rounded-xl p-4 flex items-center justify-between hover:bg-surface-bright transition-colors cursor-default">
              <div>
                <p className="text-[10px] font-bold text-slate-500 uppercase">Next 24 Hours</p>
                <h6 className="text-lg font-black text-white tabular-nums">2,985.00</h6>
              </div>
              <div className="text-right">
                <span className="inline-block px-2 py-0.5 rounded bg-secondary/10 text-secondary text-[10px] font-black uppercase mb-1">High Conviction</span>
                <p className="text-[10px] text-slate-400">84% Probability</p>
              </div>
            </div>
            
            {/* 5D Target */}
            <div className="bg-surface-container rounded-xl p-4 flex items-center justify-between hover:bg-surface-bright transition-colors cursor-default">
              <div>
                <p className="text-[10px] font-bold text-slate-500 uppercase">5-Day Outlook</p>
                <h6 className="text-lg font-black text-white tabular-nums">3,120.50</h6>
              </div>
              <div className="text-right">
                <span className="inline-block px-2 py-0.5 rounded bg-secondary/10 text-secondary text-[10px] font-black uppercase mb-1">Medium Conviction</span>
                <p className="text-[10px] text-slate-400">71% Probability</p>
              </div>
            </div>
            
            {/* 20D Target */}
            <div className="bg-surface-container rounded-xl p-4 flex items-center justify-between hover:bg-surface-bright transition-colors cursor-default">
              <div>
                <p className="text-[10px] font-bold text-slate-500 uppercase">20-Day Forecast</p>
                <h6 className="text-lg font-black text-white tabular-nums">3,450.00</h6>
              </div>
              <div className="text-right">
                <span className="inline-block px-2 py-0.5 rounded bg-primary/10 text-primary text-[10px] font-black uppercase mb-1">Speculative</span>
                <p className="text-[10px] text-slate-400">58% Probability</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Floating Quick Insights Toggle */}
      <div className="fixed bottom-8 right-8 z-50">
        <button className="flex items-center gap-2 px-6 py-3 bg-white text-background font-bold rounded-full shadow-2xl hover:scale-105 transition-transform">
          <span className="material-symbols-outlined text-lg" style={{fontVariationSettings: "'FILL' 1"}}>bolt</span>
          <span>Quick Intelligence</span>
        </button>
      </div>
    </div>
  );
};

export default Analysis;
