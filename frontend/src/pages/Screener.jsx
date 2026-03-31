import React from 'react';

const Screener = () => {
  return (
    <div className="p-8 space-y-8">
      {/* Filter Panel */}
      <section className="bg-surface-container rounded-xl p-6">
        <div className="flex flex-wrap items-center gap-8">
          <div className="flex-1 min-w-[200px] space-y-2">
            <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant/60 block">Sector Allocation</label>
            <div className="flex gap-2">
              <button className="px-4 py-1.5 rounded-md bg-primary text-on-primary text-xs font-semibold">All Sectors</button>
              <button className="px-4 py-1.5 rounded-md bg-surface-container-highest text-on-surface-variant text-xs font-medium hover:text-white transition-colors">Energy</button>
              <button className="px-4 py-1.5 rounded-md bg-surface-container-highest text-on-surface-variant text-xs font-medium hover:text-white transition-colors">IT</button>
              <button className="px-4 py-1.5 rounded-md bg-surface-container-highest text-on-surface-variant text-xs font-medium hover:text-white transition-colors">Finance</button>
            </div>
          </div>

          <div className="w-48 space-y-2">
            <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant/60 block">Market Cap</label>
            <select className="w-full bg-surface-container-highest border-none text-xs rounded-md py-2 focus:ring-1 focus:ring-primary/30 text-white">
              <option>Large Cap (&gt;₹50k Cr)</option>
              <option>Mid Cap</option>
              <option>Small Cap</option>
            </select>
          </div>

          <div className="w-48 space-y-2">
            <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant/60 block">Forecast Horizon</label>
            <div className="flex bg-surface-container-highest p-1 rounded-md">
              <button className="flex-1 text-[10px] font-bold py-1 bg-surface-container-low text-white rounded">1D</button>
              <button className="flex-1 text-[10px] font-bold py-1 text-on-surface-variant">5D</button>
              <button className="flex-1 text-[10px] font-bold py-1 text-on-surface-variant">20D</button>
            </div>
          </div>

          <div className="flex items-end h-full pt-6">
            <button className="flex items-center gap-2 text-primary text-xs font-bold hover:underline transition-all">
              <span className="material-symbols-outlined text-sm">restart_alt</span>
              Reset Filters
            </button>
          </div>
        </div>
      </section>

      {/* Main Data Table Container */}
      <section className="space-y-4">
        <div className="flex justify-between items-end">
          <div>
            <h3 className="text-2xl font-bold font-headline text-white">Nifty 50 Momentum</h3>
            <p className="text-sm text-on-surface-variant/60">Institutional-grade forecasting for major indices</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-on-surface-variant bg-surface-container-highest px-2 py-1 rounded">Live Updates: Every 60s</span>
            <button className="material-symbols-outlined text-on-surface-variant hover:text-white transition-colors">download</button>
          </div>
        </div>

        {/* Custom Table Layout */}
        <div className="bg-surface-container rounded-xl overflow-hidden">
          <div className="grid grid-cols-[2fr_1fr_1fr_1fr_1fr_1fr_1fr_1.2fr] px-6 py-4 bg-surface-container-low border-b border-white/5 text-[10px] font-black uppercase tracking-widest text-on-surface-variant/50">
            <div>Stock Instrument</div>
            <div className="text-right">Price</div>
            <div className="text-right">24h Chg</div>
            <div className="text-right">7D Trend</div>
            <div className="text-center">1D Forecast</div>
            <div className="text-center">5D Forecast</div>
            <div className="text-center">20D Forecast</div>
            <div className="text-right">Actions</div>
          </div>

          {/* Table Rows */}
          <div className="divide-y divide-white/5">
            {/* Row 1 */}
            <div className="grid grid-cols-[2fr_1fr_1fr_1fr_1fr_1fr_1fr_1.2fr] px-6 py-5 items-center hover:bg-surface-bright transition-colors group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded bg-surface-container-highest flex items-center justify-center font-bold text-primary-fixed-dim">RE</div>
                <div>
                  <h4 className="text-sm font-bold text-white group-hover:text-primary transition-colors">Reliance Industries</h4>
                  <span className="text-[10px] text-slate-500 font-medium tracking-tight">ENERGY • NSE: RELIANCE</span>
                </div>
              </div>
              <div className="text-right tabular-nums font-bold text-sm tracking-tight text-white">₹2,945.20</div>
              <div className="text-right tabular-nums text-sm font-bold text-secondary">+1.45%</div>
              <div className="flex justify-end items-center pr-2">
                <svg className="w-16 h-8 text-secondary opacity-60" fill="none" viewBox="0 0 100 40">
                  <path d="M0 35 L10 32 L20 38 L30 25 L40 28 L50 15 L60 22 L70 10 L80 18 L90 5 L100 8" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                </svg>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+0.8%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+3.2%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+8.5%</span>
              </div>
              <div className="flex justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="p-1.5 bg-surface-container-highest rounded-lg text-slate-400 hover:text-white hover:bg-primary-container/20 transition-all">
                  <span className="material-symbols-outlined text-lg">bookmark_add</span>
                </button>
                <button className="px-3 py-1.5 bg-primary-container/10 text-primary text-[10px] font-bold rounded-lg border border-primary/20 hover:bg-primary-container hover:text-white transition-all">
                  ANALYZE
                </button>
              </div>
            </div>

            {/* Row 2 */}
            <div className="grid grid-cols-[2fr_1fr_1fr_1fr_1fr_1fr_1fr_1.2fr] px-6 py-5 items-center hover:bg-surface-bright transition-colors group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded bg-surface-container-highest flex items-center justify-center font-bold text-primary-fixed-dim">TC</div>
                <div>
                  <h4 className="text-sm font-bold text-white group-hover:text-primary transition-colors">TCS Ltd</h4>
                  <span className="text-[10px] text-slate-500 font-medium tracking-tight">IT • NSE: TCS</span>
                </div>
              </div>
              <div className="text-right tabular-nums font-bold text-sm tracking-tight text-white">₹4,120.55</div>
              <div className="text-right tabular-nums text-sm font-bold text-tertiary">-0.82%</div>
              <div className="flex justify-end items-center pr-2">
                <svg className="w-16 h-8 text-tertiary opacity-60" fill="none" viewBox="0 0 100 40">
                  <path d="M0 10 L10 15 L20 12 L30 25 L40 22 L50 35 L60 30 L70 38 L80 32 L90 35 L100 30" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                </svg>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-tertiary/10 text-tertiary text-[10px] font-bold rounded">-0.2%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+1.1%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+4.2%</span>
              </div>
              <div className="flex justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="p-1.5 bg-surface-container-highest rounded-lg text-slate-400 hover:text-white hover:bg-primary-container/20 transition-all">
                  <span className="material-symbols-outlined text-lg">bookmark_add</span>
                </button>
                <button className="px-3 py-1.5 bg-primary-container/10 text-primary text-[10px] font-bold rounded-lg border border-primary/20 hover:bg-primary-container hover:text-white transition-all">
                  ANALYZE
                </button>
              </div>
            </div>

            {/* Row 3 */}
            <div className="grid grid-cols-[2fr_1fr_1fr_1fr_1fr_1fr_1fr_1.2fr] px-6 py-5 items-center hover:bg-surface-bright transition-colors group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded bg-surface-container-highest flex items-center justify-center font-bold text-primary-fixed-dim">HD</div>
                <div>
                  <h4 className="text-sm font-bold text-white group-hover:text-primary transition-colors">HDFC Bank</h4>
                  <span className="text-[10px] text-slate-500 font-medium tracking-tight">FINANCE • NSE: HDFCBANK</span>
                </div>
              </div>
              <div className="text-right tabular-nums font-bold text-sm tracking-tight text-white">₹1,540.10</div>
              <div className="text-right tabular-nums text-sm font-bold text-secondary">+2.10%</div>
              <div className="flex justify-end items-center pr-2">
                <svg className="w-16 h-8 text-secondary opacity-60" fill="none" viewBox="0 0 100 40">
                  <path d="M0 38 L15 30 L30 35 L45 20 L60 25 L75 10 L100 5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                </svg>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+1.5%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+5.8%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded">+12.4%</span>
              </div>
              <div className="flex justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="p-1.5 bg-surface-container-highest rounded-lg text-slate-400 hover:text-white hover:bg-primary-container/20 transition-all">
                  <span className="material-symbols-outlined text-lg">bookmark_add</span>
                </button>
                <button className="px-3 py-1.5 bg-primary-container/10 text-primary text-[10px] font-bold rounded-lg border border-primary/20 hover:bg-primary-container hover:text-white transition-all">
                  ANALYZE
                </button>
              </div>
            </div>

            {/* Row 4 */}
            <div className="grid grid-cols-[2fr_1fr_1fr_1fr_1fr_1fr_1fr_1.2fr] px-6 py-5 items-center hover:bg-surface-bright transition-colors group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded bg-surface-container-highest flex items-center justify-center font-bold text-primary-fixed-dim">IN</div>
                <div>
                  <h4 className="text-sm font-bold text-white group-hover:text-primary transition-colors">Infosys Ltd</h4>
                  <span className="text-[10px] text-slate-500 font-medium tracking-tight">IT • NSE: INFY</span>
                </div>
              </div>
              <div className="text-right tabular-nums font-bold text-sm tracking-tight text-white">₹1,678.90</div>
              <div className="text-right tabular-nums text-sm font-bold text-secondary">+0.12%</div>
              <div className="flex justify-end items-center pr-2">
                <svg className="w-16 h-8 text-on-surface-variant opacity-40" fill="none" viewBox="0 0 100 40">
                  <path d="M0 20 L20 20 L40 18 L60 22 L80 20 L100 20" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                </svg>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-tertiary/10 text-tertiary text-[10px] font-bold rounded">-0.4%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-tertiary/10 text-tertiary text-[10px] font-bold rounded">-1.2%</span>
              </div>
              <div className="flex justify-center">
                <span className="px-2 py-1 bg-tertiary/10 text-tertiary text-[10px] font-bold rounded">-3.5%</span>
              </div>
              <div className="flex justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="p-1.5 bg-surface-container-highest rounded-lg text-slate-400 hover:text-white hover:bg-primary-container/20 transition-all">
                  <span className="material-symbols-outlined text-lg">bookmark_add</span>
                </button>
                <button className="px-3 py-1.5 bg-primary-container/10 text-primary text-[10px] font-bold rounded-lg border border-primary/20 hover:bg-primary-container hover:text-white transition-all">
                  ANALYZE
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Intelligence Overview Grid */}
      <section className="grid grid-cols-3 gap-6">
        <div className="bg-surface-container p-6 rounded-xl space-y-4">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-primary">psychology</span>
            <h5 className="text-sm font-bold text-white uppercase tracking-tighter">Confidence Metrics</h5>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-xs text-on-surface-variant">Model Accuracy (Nifty 50)</span>
              <span className="text-xs font-bold tabular-nums">84.2%</span>
            </div>
            <div className="w-full bg-surface-container-highest h-1 rounded-full overflow-hidden">
              <div className="bg-primary h-full w-[84%]"></div>
            </div>
            <p className="text-[10px] text-on-surface-variant/50 leading-relaxed italic">Confidence index is calculated based on historical lookback and volatility skew convergence.</p>
          </div>
        </div>

        <div className="bg-surface-container p-6 rounded-xl space-y-4">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-secondary">verified_user</span>
            <h5 className="text-sm font-bold text-white uppercase tracking-tighter">Market Sentiment</h5>
          </div>
          <div className="flex items-end justify-between">
            <div className="space-y-1">
              <p className="text-3xl font-black text-white tabular-nums tracking-tighter">BULLISH</p>
              <p className="text-[10px] text-secondary font-bold">FEAR &amp; GREED INDEX: 68</p>
            </div>
            <span className="material-symbols-outlined text-5xl text-secondary opacity-20">trending_up</span>
          </div>
        </div>

        <div className="bg-surface-container p-6 rounded-xl space-y-4 border border-primary/10">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-primary-fixed-dim">bolt</span>
            <h5 className="text-sm font-bold text-white uppercase tracking-tighter">Quick Forecast Insights</h5>
          </div>
          <ul className="space-y-3">
            <li className="flex items-start gap-2">
              <span className="w-1 h-1 rounded-full bg-primary mt-1.5"></span>
              <p className="text-[11px] text-on-surface-variant">Energy sector exhibits a <span className="text-white font-bold">12.5%</span> convergence over 20D.</p>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1 h-1 rounded-full bg-primary mt-1.5"></span>
              <p className="text-[11px] text-on-surface-variant">TCS showing <span className="text-tertiary font-bold">negative divergence</span> on short-term horizons.</p>
            </li>
          </ul>
        </div>
      </section>

      {/* Floating Action Button */}
      <div className="fixed bottom-8 right-8 z-50">
        <button className="flex items-center gap-3 bg-gradient-to-br from-primary to-primary-container text-on-primary px-6 py-4 rounded-xl shadow-2xl hover:scale-105 transition-all group">
          <span className="material-symbols-outlined" style={{fontVariationSettings: "'FILL' 1"}}>add_chart</span>
          <span className="font-headline font-bold text-sm">Add New Instrument</span>
        </button>
      </div>
    </div>
  );
};

export default Screener;
