import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import { Activity } from 'lucide-react';

const TrendChart = ({ data, horizon, onHorizonChange, predictionData, ticker, availableStocks = [], onTickerChange }) => {
  return (
    <div className="xl:col-span-8 bg-surface-container rounded-2xl p-6 relative overflow-hidden animate-fade-in" style={{ animationDelay: '0.2s' }}>
      
      {/* Header */}
      <div className="flex justify-between items-start mb-6 relative z-10">
        <div>
          <div className="flex items-center gap-3">
            <h3 className="text-lg font-semibold text-on-surface tracking-tight">Market Trajectory</h3>
          </div>
          <p className="text-sm text-on-surface-variant/50 mt-1">Projected price action bounds</p>
        </div>

        <div className="flex items-center gap-3">
          {/* Stock Selector */}
          {availableStocks.length > 0 && (
            <div className="flex bg-surface-container-high p-1 rounded-lg overflow-x-auto max-w-[320px]">
              {availableStocks.map((s) => (
                <button 
                  key={s}
                  onClick={() => onTickerChange(s)}
                  className={`px-3 py-1.5 text-[11px] font-semibold rounded-md tracking-wide transition-all whitespace-nowrap ${
                    ticker === s 
                      ? 'bg-primary/15 text-primary' 
                      : 'text-on-surface-variant/60 hover:text-on-surface'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          {/* Horizon Selector */}
          <div className="flex bg-surface-container-high p-1 rounded-lg">
            {['1d', '5d', '20d'].map((h) => (
              <button 
                key={h}
                onClick={() => onHorizonChange(h)}
                className={`px-4 py-1.5 text-xs font-semibold rounded-md uppercase tracking-wider transition-all ${
                  horizon === h 
                    ? 'bg-primary text-on-primary' 
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-[320px] w-full relative z-10">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8AB4F8" stopOpacity={0.15}/>
                <stop offset="95%" stopColor="#8AB4F8" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#95DEA9" stopOpacity={0.15}/>
                <stop offset="95%" stopColor="#95DEA9" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} horizontal={true} stroke="#424750" opacity={0.15} />
            <XAxis 
              dataKey="date" 
              stroke="#8D919B" 
              fontSize={11} 
              tickLine={false} 
              axisLine={false} 
              dy={10} 
              opacity={0.6}
            />
            <YAxis 
              domain={['auto', 'auto']} 
              stroke="#8D919B" 
              fontSize={11} 
              tickLine={false} 
              axisLine={false} 
              orientation="right" 
              tickFormatter={(v) => v.toLocaleString()} 
              dx={10}
              opacity={0.6}
            />
            
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#2A2A2A', 
                borderColor: '#424750', 
                borderRadius: '12px', 
                color: '#E5E2E1' 
              }}
              itemStyle={{ color: '#E5E2E1', fontSize: '13px', fontWeight: '500' }}
              labelStyle={{ color: '#9AA0A6', fontSize: '11px', marginBottom: '6px' }}
            />
            
            {/* Historical Price */}
            <Area 
              type="monotone" 
              dataKey="price" 
              stroke="#8AB4F8" 
              strokeWidth={2} 
              fillOpacity={1} 
              fill="url(#colorPrice)" 
              activeDot={{ r: 5, fill: '#8AB4F8', stroke: '#131313', strokeWidth: 2 }} 
            />
            
            {/* Predicted Path */}
            <Area 
              type="monotone" 
              dataKey="predicted" 
              stroke="#95DEA9" 
              strokeWidth={2} 
              strokeDasharray="6 4" 
              fillOpacity={1} 
              fill="url(#colorPred)" 
              activeDot={{ r: 5, fill: '#95DEA9' }} 
            />
            
            {data.length > 0 && (
              <ReferenceLine 
                x={data[Math.max(0, data.length - 2)]?.date} 
                stroke="#424750" 
                strokeDasharray="3 3" 
                opacity={0.4} 
              />
            )}
          </AreaChart>
        </ResponsiveContainer>

        {/* Target overlay */}
        {predictionData && (
          <div className="absolute top-3 right-4 bg-surface-container-high rounded-xl p-4 max-w-[180px] animate-fade-in">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-3.5 h-3.5 text-gain" />
              <span className="text-[10px] font-semibold text-on-surface-variant/50 uppercase tracking-widest">Target ({horizon})</span>
            </div>
            <div className="text-xl font-semibold text-on-surface tabular-nums tracking-tight">
              ₹{predictionData.targetPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </div>
            <div className={`text-sm font-semibold tabular-nums mt-1 ${predictionData.expectedReturn >= 0 ? 'text-gain' : 'text-loss'}`}>
              {predictionData.expectedReturn >= 0 ? '+' : ''}{(predictionData.expectedReturn * 100).toFixed(2)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrendChart;
