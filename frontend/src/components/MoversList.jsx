import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { motion } from 'framer-motion';

const MoverRow = ({ isGainer, stock, sector, price, change, index }) => (
  <motion.div 
    initial={{ opacity: 0, x: -10 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ delay: index * 0.1 }}
    className="flex items-center justify-between p-3.5 rounded-xl bg-white/5 border border-white/5 hover:border-white/20 transition-all group cursor-pointer hover:bg-white/10"
  >
    <div className="flex items-center gap-4">
      <div className={`w-10 h-10 rounded-xl ${isGainer ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'} flex items-center justify-center border ${isGainer ? 'border-emerald-500/30' : 'border-rose-500/30'}`}>
        {isGainer 
          ? <TrendingUp className="w-5 h-5" /> 
          : <TrendingDown className="w-5 h-5" />
        }
      </div>
      <div>
        <div className="text-sm font-headline font-bold text-slate-100 max-w-[140px] truncate">{stock}</div>
        <div className="text-[11px] font-body text-slate-500 max-w-[140px] truncate">{sector || 'Equities'}</div>
      </div>
    </div>
    <div className="text-right tabular-nums">
      <div className="text-sm font-semibold text-white">₹{price}</div>
      <div className={`text-xs font-bold ${isGainer ? 'text-emerald-400' : 'text-rose-400'}`}>
        {isGainer ? '+' : ''}{change}
      </div>
    </div>
  </motion.div>
);

const MoversList = ({ predictions = [], title = "Top Movers", isGainer = true }) => {
  // Sort by expected return
  const sorted = [...predictions].sort((a, b) => {
    const retA = a.prediction?.predicted_return || 0;
    const retB = b.prediction?.predicted_return || 0;
    return isGainer ? retB - retA : retA - retB;
  }).slice(0, 3);

  return (
    <div className="bg-black/40 border border-white/10 rounded-2xl p-6 backdrop-blur-xl">
      <div className="flex justify-between items-center mb-6">
        <h3 className="font-headline font-bold text-base text-slate-100 tracking-tight">{title}</h3>
        <button className="text-[11px] font-bold text-blue-400 hover:text-blue-300 transition-colors uppercase tracking-wider">View All</button>
      </div>
      
      <div className="space-y-3">
        {sorted.length > 0 ? (
          sorted.map((item, idx) => (
            <MoverRow 
              key={`${item.ticker}-${idx}`}
              index={idx}
              isGainer={isGainer}
              stock={item.ticker.replace('_NS', '')}
              sector={item.sector}
              price={item.current_price?.toFixed(2) || '0.00'}
              change={`${(item.prediction?.predicted_return * 100 || 0).toFixed(2)}%`}
            />
          ))
        ) : (
          <div className="text-xs font-body text-slate-500 p-6 text-center bg-white/5 rounded-xl border border-white/5 animate-pulse">
            Scanning signals...
          </div>
        )}
      </div>
    </div>
  );
};

export default MoversList;
