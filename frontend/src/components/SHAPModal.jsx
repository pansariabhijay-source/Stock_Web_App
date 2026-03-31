import React, { useState, useEffect } from 'react';
import { X, Bot, Zap, ArrowRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
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
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="absolute inset-0 bg-[#070A11]/80 backdrop-blur-md"
        />

        {/* Modal Content */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative w-full max-w-3xl bg-[#0B0F19]/90 border border-white/10 rounded-3xl shadow-[0_0_100px_rgba(59,130,246,0.15)] overflow-hidden flex flex-col max-h-[85vh]"
        >
          {/* Top Decorative Glow */}
          <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-50" />
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/20 blur-[100px] rounded-full" />
          
          <div className="flex justify-between items-start p-8 border-b border-white/5 relative z-10">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="bg-blue-500/20 p-2 rounded-xl border border-blue-500/30 text-blue-400">
                  <Bot size={20} />
                </div>
                <h2 className="text-2xl font-headline font-black text-white tracking-tight">
                  {stock.ticker.replace('_NS', '')}
                </h2>
              </div>
              <p className="text-sm font-body text-slate-400">
                AI Intelligence Report • {stock.horizon} Horizon
              </p>
            </div>
            <button onClick={onClose} className="p-2 bg-white/5 hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 rounded-full transition-colors border border-transparent hover:border-rose-500/30">
              <X size={20} />
            </button>
          </div>

          <div className="p-8 overflow-y-auto flex-1 relative z-10 custom-scrollbar">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 space-y-4">
                <div className="w-12 h-12 rounded-full border-4 border-white/5 border-t-blue-500 animate-spin" />
                <div className="text-sm font-headline text-slate-500 animate-pulse tracking-widest uppercase">
                  Synthesizing Neural Pathways...
                </div>
              </div>
            ) : explainData ? (
              <div className="space-y-10">
                {/* Interpretation Hero */}
                <div className="bg-gradient-to-br from-blue-900/40 to-indigo-900/20 border border-blue-500/20 p-6 rounded-2xl relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-1 h-full bg-blue-500" />
                  <div className="flex items-start gap-4">
                    <Zap className="text-blue-400 mt-1 flex-shrink-0" size={24} />
                    <p className="text-base font-body text-slate-200 leading-relaxed">
                      {explainData.interpretation}
                    </p>
                  </div>
                </div>

                {/* Factors Chart */}
                <div>
                  <h3 className="font-headline font-bold text-sm text-slate-400 mb-6 flex items-center gap-2 uppercase tracking-wider">
                    <ArrowRight size={16} /> Key Predictive Drivers
                  </h3>
                  
                  <div className="space-y-4">
                    {explainData.top_features.map((feat, i) => {
                      const isPos = feat.direction === 'positive';
                      const label = feat.feature.replace('target_', '').replace(/_/g, ' ').toUpperCase();
                      const barWidth = Math.max(Math.min(Math.abs(feat.importance) * 15000, 100), 5); // visually scale SHAP
                      
                      return (
                        <div key={i} className="flex items-center gap-6 group">
                          <div className="w-1/4 text-xs font-bold text-slate-300 truncate text-right group-hover:text-white transition-colors">
                            {label}
                          </div>
                          
                          {/* Zero Axis Chart */}
                          <div className="flex-1 flex items-center relative h-8 bg-black/20 rounded-lg p-1 border border-white/5">
                            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700/50 z-10" />
                            
                            <div className="w-1/2 h-full flex justify-end items-center pr-1">
                              {!isPos && (
                                <motion.div 
                                  initial={{ width: 0 }} animate={{ width: `${barWidth}%` }}
                                  className="h-full bg-gradient-to-l from-rose-500 to-rose-600 rounded-sm relative"
                                >
                                  <div className="absolute inset-x-0 top-0 h-px bg-white/20" />
                                </motion.div>
                              )}
                            </div>
                            
                            <div className="w-1/2 h-full flex justify-start items-center pl-1">
                              {isPos && (
                                <motion.div 
                                  initial={{ width: 0 }} animate={{ width: `${barWidth}%` }}
                                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-sm relative"
                                >
                                  <div className="absolute inset-x-0 top-0 h-px bg-white/20" />
                                </motion.div>
                              )}
                            </div>
                          </div>

                          <div className="w-16 text-[10px] tabular-nums font-mono text-slate-500 text-right">
                            {feat.importance.toFixed(4)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            ) : (
              <div className="text-center font-body text-slate-500 py-12 bg-white/5 rounded-2xl border border-white/5">
                Model introspection unavailable
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default SHAPModal;
