import React, { useState, useEffect } from 'react';
import { X, Bot, Zap, ArrowRight } from 'lucide-react';
import { api } from '../api';

const SHAPModal = ({ stock, onClose }) => {
  const [explainData, setExplainData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchExplain = async () => {
      setLoading(true);
      try {
        const res = await api.explainPrediction(stock.ticker, stock.horizon);
        setExplainData(res);
      } catch (err) {
        console.error("Explain error", err);
      } finally {
        setLoading(false);
      }
    };
    if (stock) fetchExplain();
  }, [stock]);

  if (!stock) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div onClick={onClose} className="absolute inset-0 bg-background/80" />

      {/* Modal */}
      <div className="relative w-full max-w-3xl bg-surface-container-high rounded-2xl overflow-hidden flex flex-col max-h-[85vh] shadow-[0_8px_64px_rgba(0,0,0,0.5)] animate-fade-in">
        
        {/* Header */}
        <div className="flex justify-between items-start p-6 border-b border-outline-variant/20 relative z-10">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="bg-primary/10 p-2 rounded-lg text-primary">
                <Bot size={18} />
              </div>
              <h2 className="text-xl font-semibold text-on-surface tracking-tight">
                {stock.ticker.replace('_NS', '')}
              </h2>
            </div>
            <p className="text-sm text-on-surface-variant/50 ml-[44px]">
              AI Intelligence Report • {stock.horizon} Horizon
            </p>
          </div>
          <button 
            onClick={onClose} 
            className="p-2 text-on-surface-variant hover:text-error hover:bg-error/10 rounded-lg transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1 relative z-10">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16 space-y-4">
              <div className="w-10 h-10 rounded-full border-3 border-surface-container-highest border-t-primary animate-spin" />
              <div className="text-sm text-on-surface-variant/50 animate-pulse">
                Analyzing model features...
              </div>
            </div>
          ) : explainData ? (
            <div className="space-y-8">
              {/* Interpretation */}
              <div className="bg-primary/5 p-5 rounded-xl relative">
                <div className="absolute top-0 left-0 w-1 h-full bg-primary rounded-l-xl" />
                <div className="flex items-start gap-3 ml-2">
                  <Zap className="text-primary mt-0.5 flex-shrink-0" size={18} />
                  <p className="text-sm text-on-surface leading-relaxed">
                    {explainData.interpretation}
                  </p>
                </div>
              </div>

              {/* Factors */}
              <div>
                <h3 className="text-[11px] font-semibold text-on-surface-variant/50 mb-5 flex items-center gap-2 uppercase tracking-widest">
                  <ArrowRight size={14} /> Key Predictive Drivers
                </h3>
                
                <div className="space-y-3">
                  {explainData.top_features.map((feat, i) => {
                    const isPos = feat.direction === 'positive';
                    const label = feat.feature.replace('target_', '').replace(/_/g, ' ').toUpperCase();
                    const barWidth = Math.max(Math.min(Math.abs(feat.importance) * 15000, 100), 5);
                    
                    return (
                      <div key={i} className="flex items-center gap-4 group">
                        <div className="w-1/4 text-[11px] font-medium text-on-surface-variant truncate text-right group-hover:text-on-surface transition-colors">
                          {label}
                        </div>
                        
                        <div className="flex-1 flex items-center relative h-6 bg-surface-container rounded-md p-0.5">
                          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-outline-variant/20 z-10" />
                          
                          <div className="w-1/2 h-full flex justify-end items-center pr-0.5">
                            {!isPos && (
                              <div 
                                className="h-full bg-error/60 rounded-sm transition-all duration-700"
                                style={{ width: `${barWidth}%` }}
                              />
                            )}
                          </div>
                          
                          <div className="w-1/2 h-full flex justify-start items-center pl-0.5">
                            {isPos && (
                              <div 
                                className="h-full bg-tertiary/60 rounded-sm transition-all duration-700"
                                style={{ width: `${barWidth}%` }}
                              />
                            )}
                          </div>
                        </div>

                        <div className="w-14 text-[10px] tabular-nums font-mono text-on-surface-variant/50 text-right">
                          {feat.importance.toFixed(4)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-on-surface-variant/50 py-12 bg-surface-container rounded-xl text-sm">
              Model introspection unavailable
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SHAPModal;
