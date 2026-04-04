import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

const MoverRow = ({ stock, sector, price, change, isGainer }) => {
  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-surface-container-high hover:bg-surface-bright transition-colors cursor-pointer">
      <div className="flex items-center gap-3">
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center text-sm ${
          isGainer ? 'bg-gain/10 text-gain' : 'bg-loss/10 text-loss'
        }`}>
          {isGainer ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
        </div>
        <div>
          <div className="text-sm font-semibold text-on-surface max-w-[130px] truncate">{stock}</div>
          <div className="text-[11px] text-on-surface-variant/50 max-w-[130px] truncate">{sector || 'Equities'}</div>
        </div>
      </div>
      <div className="text-right tabular-nums">
        <div className="text-sm font-medium text-on-surface">₹{price}</div>
        <div className={`text-xs font-semibold ${isGainer ? 'text-gain' : 'text-loss'}`}>
          {isGainer ? '+' : '-'}{change}
        </div>
      </div>
    </div>
  );
};

const MoversList = ({ predictions = [], title = "Top Movers", isGainer = true }) => {
  const sorted = [...predictions].sort((a, b) => {
    const retA = a.prediction?.predicted_return || 0;
    const retB = b.prediction?.predicted_return || 0;
    return isGainer ? retB - retA : retA - retB;
  }).slice(0, 3);

  return (
    <div className="bg-surface-container rounded-2xl p-5">
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-semibold text-sm text-on-surface tracking-tight">{title}</h3>
        <button className="text-[11px] font-medium text-primary hover:text-primary-fixed transition-colors">View All</button>
      </div>
      
      <div className="space-y-2">
        {sorted.length > 0 ? (
          sorted.map((item, idx) => (
            <MoverRow 
              key={`${item.ticker}-${idx}`}
              isGainer={isGainer}
              stock={item.ticker.replace('_NS', '')}
              sector={item.sector}
              price={item.current_price?.toFixed(2) || '0.00'}
              change={`${Math.abs(item.prediction?.predicted_return * 100 || 0).toFixed(2)}%`}
            />
          ))
        ) : (
          <div className="text-xs text-on-surface-variant/50 p-6 text-center bg-surface-container-high rounded-lg animate-pulse">
            Scanning signals...
          </div>
        )}
      </div>
    </div>
  );
};

export default MoversList;
