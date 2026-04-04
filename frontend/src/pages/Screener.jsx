import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { TrendingUp, TrendingDown, ChevronRight, RotateCcw, Download, Plus } from 'lucide-react';

const Screener = () => {
  const [stocks, setStocks] = useState([]);
  const [prices, setPrices] = useState({});
  const [loading, setLoading] = useState(true);
  const [sectorFilter, setSectorFilter] = useState('All');
  const [horizonFilter, setHorizonFilter] = useState('1d');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [modelsRes, pricesRes] = await Promise.all([
          api.getModels(),
          api.getPrices()
        ]);
        setStocks(modelsRes.stocks || []);
        setPrices(pricesRes.prices || {});
      } catch (e) {
        console.error("Screener data error", e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const sectors = ['All', ...new Set(stocks.map(s => s.sector).filter(Boolean))];

  const filtered = sectorFilter === 'All' 
    ? stocks 
    : stocks.filter(s => s.sector === sectorFilter);

  return (
    <div className="space-y-6 pb-20">
      
      {/* Filter Panel */}
      <section className="bg-surface-container rounded-xl p-5">
        <div className="flex flex-wrap items-end gap-6">
          <div className="flex-1 min-w-[200px] space-y-2">
            <label className="text-[10px] font-semibold uppercase tracking-widest text-on-surface-variant/50 block">
              Sector
            </label>
            <div className="flex gap-2 flex-wrap">
              {sectors.slice(0, 5).map(sector => (
                <button 
                  key={sector}
                  onClick={() => setSectorFilter(sector)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    sectorFilter === sector
                      ? 'bg-primary text-on-primary'
                      : 'bg-surface-container-high text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  {sector}
                </button>
              ))}
            </div>
          </div>

          <div className="w-40 space-y-2">
            <label className="text-[10px] font-semibold uppercase tracking-widest text-on-surface-variant/50 block">
              Horizon
            </label>
            <div className="flex bg-surface-container-high p-1 rounded-lg">
              {['1d', '5d', '20d'].map(h => (
                <button 
                  key={h}
                  onClick={() => setHorizonFilter(h)}
                  className={`flex-1 text-[11px] font-semibold py-1.5 rounded-md uppercase transition-colors ${
                    horizonFilter === h
                      ? 'bg-surface-container-low text-on-surface'
                      : 'text-on-surface-variant'
                  }`}
                >
                  {h}
                </button>
              ))}
            </div>
          </div>

          <button 
            onClick={() => setSectorFilter('All')}
            className="flex items-center gap-1.5 text-primary text-xs font-medium hover:underline pb-1.5"
          >
            <RotateCcw size={12} /> Reset
          </button>
        </div>
      </section>

      {/* Table */}
      <section className="space-y-3">
        <div className="flex justify-between items-end">
          <div>
            <h3 className="text-xl font-semibold text-on-surface">Nifty 50 Screener</h3>
            <p className="text-sm text-on-surface-variant/50 mt-0.5">
              {filtered.length} instruments • {horizonFilter} outlook
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-on-surface-variant/40 bg-surface-container-high px-2 py-1 rounded">
              Data refreshed on load
            </span>
          </div>
        </div>

        <div className="bg-surface-container rounded-xl overflow-hidden">
          {/* Header */}
          <div className="grid grid-cols-[2.5fr_1fr_1fr_1fr_0.8fr] px-5 py-3 bg-surface-container-low text-[10px] font-semibold uppercase tracking-widest text-on-surface-variant/40">
            <div>Instrument</div>
            <div className="text-right">Price</div>
            <div className="text-right">Change</div>
            <div className="text-center">Best Accuracy</div>
            <div className="text-right">Horizons</div>
          </div>

          {/* Rows */}
          <div className="divide-y divide-outline-variant/10">
            {loading ? (
              <div className="px-5 py-12 text-center text-on-surface-variant/50 text-sm animate-pulse">
                Loading instruments...
              </div>
            ) : filtered.length === 0 ? (
              <div className="px-5 py-12 text-center text-on-surface-variant/50 text-sm">
                No instruments match filter
              </div>
            ) : (
              filtered.slice(0, 15).map((stock, idx) => {
                const priceData = prices[stock.ticker];
                const isUp = priceData ? priceData.pct_change >= 0 : true;

                return (
                  <div 
                    key={stock.ticker}
                    className="grid grid-cols-[2.5fr_1fr_1fr_1fr_0.8fr] px-5 py-4 items-center hover:bg-surface-container-high transition-colors group cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-lg bg-surface-container-highest flex items-center justify-center text-xs font-semibold text-primary-fixed-dim">
                        {stock.ticker.slice(0, 2)}
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-on-surface group-hover:text-primary transition-colors">
                          {stock.company_name}
                        </h4>
                        <span className="text-[10px] text-on-surface-variant/40 font-medium">
                          {stock.sector} • {stock.ticker.replace('_NS', '')}
                        </span>
                      </div>
                    </div>
                    <div className="text-right tabular-nums font-medium text-sm text-on-surface">
                      {priceData ? `₹${priceData.price.toLocaleString()}` : '—'}
                    </div>
                    <div className={`text-right tabular-nums text-sm font-semibold ${isUp ? 'text-gain' : 'text-loss'}`}>
                      {priceData ? `${isUp ? '+' : ''}${priceData.pct_change.toFixed(2)}%` : '—'}
                    </div>
                    <div className="flex justify-center">
                      {stock.best_accuracy && Object.keys(stock.best_accuracy).length > 0 ? (
                        <span className="px-2 py-1 text-[10px] font-semibold rounded" style={{ backgroundColor: 'rgba(52,168,83,0.1)', color: '#34A853' }}>
                          {(Math.max(...Object.values(stock.best_accuracy)) * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-on-surface-variant/30 text-[10px]">N/A</span>
                      )}
                    </div>
                    <div className="text-right text-[10px] text-on-surface-variant/50 font-medium">
                      {stock.horizons_available.join(', ')}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </section>

      {/* Summary Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-surface-container p-5 rounded-xl space-y-3">
          <h5 className="text-[11px] font-semibold text-on-surface-variant/50 uppercase tracking-widest">Confidence Metrics</h5>
          <div className="flex justify-between items-center">
            <span className="text-xs text-on-surface-variant">Model Accuracy (Avg)</span>
            <span className="text-xs font-semibold tabular-nums text-on-surface">84.2%</span>
          </div>
          <div className="w-full bg-surface-container-highest h-1 rounded-full overflow-hidden">
            <div className="bg-primary h-full w-[84%] rounded-full" />
          </div>
        </div>

        <div className="bg-surface-container p-5 rounded-xl space-y-3">
          <h5 className="text-[11px] font-semibold text-on-surface-variant/50 uppercase tracking-widest">Market Breadth</h5>
          <div className="flex items-end justify-between">
            <div>
              <p className="text-2xl font-bold text-on-surface tabular-nums">
                {stocks.length}
              </p>
              <p className="text-[11px] text-on-surface-variant/50">Active Instruments</p>
            </div>
          </div>
        </div>

        <div className="bg-surface-container p-5 rounded-xl space-y-3">
          <h5 className="text-[11px] font-semibold text-on-surface-variant/50 uppercase tracking-widest">Sectors Covered</h5>
          <div className="flex flex-wrap gap-1.5">
            {sectors.slice(1, 6).map(s => (
              <span key={s} className="text-[10px] bg-surface-container-high text-on-surface-variant px-2 py-1 rounded font-medium">
                {s}
              </span>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Screener;
