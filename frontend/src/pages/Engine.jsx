import React from 'react';

const Engine = () => {
  return (
    <div className="max-w-7xl mx-auto px-12 py-16">
      {/* Hero Section */}
      <section className="mb-24">
        <div className="inline-flex items-center px-3 py-1 rounded-full bg-primary-container/10 border border-primary-container/20 text-primary text-xs font-bold uppercase tracking-widest mb-6">
          Technical Whitepaper
        </div>
        <h1 className="text-6xl font-headline font-extrabold tracking-tighter text-on-surface mb-6 max-w-3xl">
          The Engine: High-Conviction <span className="text-primary italic">Intelligence.</span>
        </h1>
        <p className="text-xl text-on-surface-variant max-w-2xl leading-relaxed">
          A deep dive into the Stratos Terminal backend. Explore how we combine ensemble learning, asynchronous processing, and institutional-grade deployment to deliver market signals.
        </p>
      </section>

      {/* Architecture Diagram Grid */}
      <section className="grid grid-cols-12 gap-8 mb-32 items-start">
        <div className="col-span-12 lg:col-span-7">
          <div className="relative aspect-video rounded-3xl overflow-hidden bg-surface-container group">
            <img alt="Architecture Diagram" className="w-full h-full object-cover opacity-60 transition-transform duration-700 group-hover:scale-105" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBnvmjNrKSLBPsDRlHPT4QuiMSQ4afi9ilECJEaZ6CQEZP1GP6byrpGb9TiADVBhJfF_e6amyMXJFM9i7SXmkeaHa-M8Pkq3bp15bsDLtuFSVig1hWdWI4h-7pTj-qOHdXzGt2zfbQIhX0kXGUteWTDmwmPUjcF--eWcATynOqnL3IRYQARpKy-53zWeBxO8OADIbanjesQMBMjzzO2mOv-Ixc3h0VaMHdcGaBgGYdhyEfjZygKKKCAPLTMd6eaCgfMKCyeh4BOojlh" />
            <div className="absolute inset-0 bg-gradient-to-tr from-background via-transparent to-primary/10"></div>
            
            {/* Floating Glass Labels */}
            <div className="absolute top-10 left-10 backdrop-blur-md bg-surface-variant/60 p-4 rounded-xl border border-white/5">
              <span className="block text-[10px] text-primary font-bold uppercase mb-1">Inference Layer</span>
              <span className="text-sm font-semibold text-white">XGBoost &amp; LightGBM</span>
            </div>
            <div className="absolute bottom-10 right-10 backdrop-blur-md bg-surface-variant/60 p-4 rounded-xl border border-white/5 text-right">
              <span className="block text-[10px] text-secondary font-bold uppercase mb-1">Pipeline State</span>
              <span className="text-sm font-semibold text-white">Render Cloud Managed</span>
            </div>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-5 space-y-8">
          <div className="p-8 rounded-3xl bg-surface-container-low border border-outline-variant/10">
            <h3 className="text-2xl font-headline font-bold text-white mb-4">Neural Data Pipelines</h3>
            <p className="text-on-surface-variant leading-relaxed">
              Our proprietary data ingestion layer processes over 40,000 tickers per second. Unlike retail platforms, we utilize raw FIX protocols to minimize latency before feature engineering.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-6 rounded-3xl bg-surface-container border border-outline-variant/5">
              <span className="material-symbols-outlined text-primary mb-3">bolt</span>
              <div className="text-2xl font-headline font-bold text-white tabular-nums">42ms</div>
              <div className="text-xs text-slate-500 uppercase font-bold tracking-tighter">API Latency</div>
            </div>
            <div className="p-6 rounded-3xl bg-surface-container border border-outline-variant/5">
              <span className="material-symbols-outlined text-secondary mb-3">verified</span>
              <div className="text-2xl font-headline font-bold text-white tabular-nums">99.9%</div>
              <div className="text-xs text-slate-500 uppercase font-bold tracking-tighter">Uptime SLA</div>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Specs Bento */}
      <section className="mb-32">
        <h2 className="text-3xl font-headline font-bold text-white mb-12 flex items-center gap-4">
          <span className="w-8 h-[2px] bg-primary"></span>
          Technical Specifications
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* FastAPI */}
          <div className="col-span-1 p-8 rounded-3xl bg-surface-container-high relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 text-white/5 text-8xl material-symbols-outlined">api</div>
            <h3 className="text-xl font-headline font-bold text-white mb-4">FastAPI Microservices</h3>
            <p className="text-on-surface-variant text-sm leading-relaxed mb-6">
              High-performance Python 3.11 backend leveraging Pydantic V2 for data validation. Asynchronous I/O ensures concurrent data streaming across terminal modules.
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="px-2 py-1 rounded-md bg-white/5 text-[10px] font-mono text-primary">asyncio</span>
              <span className="px-2 py-1 rounded-md bg-white/5 text-[10px] font-mono text-primary">uvicorn</span>
            </div>
          </div>

          {/* Ensemble ML */}
          <div className="col-span-1 md:col-span-2 p-8 rounded-3xl bg-surface-container relative overflow-hidden border border-primary/10">
            <div className="flex flex-col md:flex-row gap-8 items-center">
              <div className="flex-1">
                <h3 className="text-xl font-headline font-bold text-white mb-4">Gradient Boosted Ensembles</h3>
                <p className="text-on-surface-variant text-sm leading-relaxed mb-6">
                  Stratos uses a hybrid weighting of XGBoost and LightGBM models. This ensemble approach reduces variance and improves prediction stability during high-volatility events. We prioritize Gini importance and coverage to prune noise.
                </p>
                <div className="flex gap-4">
                  <div className="flex items-center gap-2 text-secondary text-xs font-bold">
                    <span className="w-2 h-2 rounded-full bg-secondary"></span> LightGBM Optimized
                  </div>
                  <div className="flex items-center gap-2 text-primary text-xs font-bold">
                    <span className="w-2 h-2 rounded-full bg-primary"></span> XGBoost Robust
                  </div>
                </div>
              </div>
              <div className="w-full md:w-48 aspect-square rounded-full bg-surface-container-highest flex items-center justify-center border border-outline-variant/20 p-4">
                <div className="w-full h-full rounded-full border-4 border-dashed border-primary/20 animate-spin-slow flex items-center justify-center">
                  <span className="material-symbols-outlined text-4xl text-primary">hub</span>
                </div>
              </div>
            </div>
          </div>

          {/* Deployment */}
          <div className="col-span-1 md:col-span-2 p-8 rounded-3xl bg-surface-container-high border border-outline-variant/10">
            <div className="flex items-start justify-between mb-8">
              <div>
                <h3 className="text-xl font-headline font-bold text-white mb-2">Render Deployment Architecture</h3>
                <p className="text-on-surface-variant text-sm">Managed cloud orchestration for zero-downtime scaling.</p>
              </div>
              <span className="px-3 py-1 bg-secondary/10 text-secondary text-[10px] font-bold rounded-full border border-secondary/20">PRODUCTION READY</span>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="text-white text-xs font-bold">Auto-Scaling</div>
                <div className="h-1 bg-surface-variant rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-3/4"></div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-white text-xs font-bold">Global CDN</div>
                <div className="h-1 bg-surface-variant rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-full"></div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-white text-xs font-bold">DB Replication</div>
                <div className="h-1 bg-surface-variant rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-1/2"></div>
                </div>
              </div>
            </div>
          </div>

          {/* Intelligence Score */}
          <div className="col-span-1 p-8 rounded-3xl bg-gradient-to-br from-primary-container to-primary flex flex-col justify-end">
            <span className="material-symbols-outlined text-on-primary mb-4 text-4xl" style={{fontVariationSettings: "'FILL' 1"}}>psychology</span>
            <h3 className="text-xl font-headline font-bold text-on-primary mb-2">Intelligence Score</h3>
            <p className="text-on-primary/80 text-xs leading-relaxed">
              Our proprietary metric derived from 200+ features across sentiment, macro, and technical indicators.
            </p>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="p-16 rounded-3xl bg-surface-container-highest border border-outline-variant/20 flex flex-col items-center text-center overflow-hidden relative">
        <h2 className="text-4xl font-headline font-extrabold text-white mb-6 relative z-10">Ready to build on Stratos?</h2>
        <p className="text-on-surface-variant max-w-xl mb-10 relative z-10">
          Access our API documentation or download the full technical specification for a deep-dive into our modeling parameters.
        </p>
        <div className="flex gap-4 relative z-10">
          <button className="px-8 py-3 bg-primary text-on-primary font-bold rounded-md hover:bg-primary-fixed-dim transition-all shadow-lg shadow-primary/20">
            Explore API Docs
          </button>
          <button className="px-8 py-3 bg-surface-container border border-outline-variant/30 text-white font-bold rounded-md hover:bg-surface-bright transition-all">
            Technical PDF
          </button>
        </div>
      </section>

      {/* Footer Info */}
      <footer className="pt-12 mt-12 border-t border-outline-variant/10 text-slate-500 flex justify-between items-center">
        <div className="flex items-center gap-8">
          <span className="text-xs font-bold uppercase tracking-widest">Stratos v4.2.0-stable</span>
          <span className="text-xs font-bold uppercase tracking-widest">Latency: 14ms (London-1)</span>
        </div>
        <div className="flex gap-6 text-xs font-medium">
          <a className="hover:text-primary transition-colors" href="#">Privacy Policy</a>
          <a className="hover:text-primary transition-colors" href="#">Security Audit</a>
          <a className="hover:text-primary transition-colors" href="#">Github</a>
        </div>
      </footer>
    </div>
  );
};

export default Engine;
