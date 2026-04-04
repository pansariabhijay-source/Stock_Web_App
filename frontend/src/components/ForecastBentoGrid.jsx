import React, { useState, useEffect } from 'react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import { ArrowUpRight, ArrowDownRight, Target } from 'lucide-react';
import { api } from '../api';

const ForecastBentoGrid = ({ forecasts, onCardClick }) => {
  if (!forecasts || forecasts.length === 0) {
    return (
      <div className="col-span-1 xl:col-span-8 flex items-center justify-center p-8 bg-surface-container rounded-2xl">
        <div className="text-on-surface-variant/50 text-sm animate-pulse">Computing Predictive Horizons...</div>
      </div>
    );
  }

  return (
    <div className="col-span-1 xl:col-span-8 grid grid-cols-1 md:grid-cols-3 gap-4 animate-fade-in" style={{ animationDelay: '0.1s' }}>
      {forecasts.map((f, i) => (
        <ForecastCard key={`${f.ticker}-${i}`} data={f} onClick={() => onCardClick(f.ticker)} />
      ))}
    </div>
  );
};

const ForecastCard = ({ data, onClick }) => {
  const { ticker, prediction, current_price } = data;
  const isUp = prediction.predicted_return >= 0;
  const expectedReturn = (prediction.predicted_return * 100).toFixed(2);
  const targetPrice = (current_price * (1 + prediction.predicted_return)).toFixed(2);
  const confPercent = Math.round(prediction.probability * 100);

  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchSparklineData = async () => {
      try {
        const res = await api.getHistory(ticker, 10);
        if (res && res.history) {
          setHistory(res.history);
        }
      } catch (err) {
        console.error("Failed to fetch sparkline history for", ticker, err);
      }
    };
    fetchSparklineData();
  }, [ticker]);

  const sparkData = history.length > 0 ? [
    ...history.map(item => ({ value: item.price })),
    { value: current_price },
    { value: parseFloat(targetPrice) }
  ] : [
    { value: current_price * 0.99 },
    { value: current_price },
    { value: parseFloat(targetPrice) }
  ];

  return (
    <div 
      onClick={onClick}
      className="cursor-pointer bg-surface-container rounded-2xl p-5 transition-colors duration-200 hover:bg-surface-container-high group"
    >
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <div className="text-base font-semibold text-on-surface">{ticker.replace('_NS', '')}</div>
          <div className="text-[11px] text-on-surface-variant/50">NSE</div>
        </div>
        <div className={`px-2.5 py-1 rounded-full text-[11px] font-semibold flex items-center gap-1 ${
          isUp 
            ? "bg-gain/10 text-gain" 
            : "bg-loss/10 text-loss"
        }`}>
          {isUp ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
          {isUp ? '+' : ''}{expectedReturn}%
        </div>
      </div>

      {/* Price */}
      <div className="mb-3">
        <div className="text-2xl font-semibold text-on-surface tabular-nums tracking-tight">
          ₹{current_price.toLocaleString()}
        </div>
        <div className="text-[11px] text-on-surface-variant/50 flex items-center gap-1 mt-1">
          <Target size={11} /> Target: <span className="text-on-surface-variant font-medium">₹{targetPrice}</span>
        </div>
      </div>

      {/* Sparkline */}
      <div className="h-12 w-full mt-3 opacity-60 group-hover:opacity-100 transition-opacity">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={sparkData}>
            <defs>
              <linearGradient id={`grad-${ticker}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={isUp ? "#34A853" : "#EA4335"} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={isUp ? "#34A853" : "#EA4335"} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <YAxis domain={['dataMin', 'dataMax']} hide />
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke={isUp ? "#34A853" : "#EA4335"} 
              strokeWidth={2}
              fillOpacity={1} 
              fill={`url(#grad-${ticker})`} 
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Confidence */}
      <div className="mt-4 pt-3 border-t border-outline-variant/20 flex justify-between items-center">
        <span className="text-[11px] text-on-surface-variant/50">Confidence</span>
        <div className="w-1/2 bg-surface-container-highest rounded-full h-1 overflow-hidden">
          <div 
            className="h-full rounded-full transition-all duration-1000"
            style={{ width: `${confPercent}%`, backgroundColor: isUp ? '#34A853' : '#EA4335' }}
          />
        </div>
        <span className="text-[11px] font-semibold text-on-surface-variant tabular-nums">{confPercent}%</span>
      </div>
    </div>
  );
};

export default ForecastBentoGrid;
