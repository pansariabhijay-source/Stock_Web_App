import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine 
} from 'recharts';
import { Activity } from 'lucide-react';
import { motion } from 'framer-motion';

const TrendChart = ({ data, horizon, onHorizonChange, predictionData }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="lg:col-span-8 bg-black/40 border border-white/10 rounded-2xl p-8 backdrop-blur-xl shadow-2xl relative overflow-hidden"
    >
      {/* Background visual flair */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-[120px] -z-10 pointer-events-none" />

      <div className="flex justify-between items-start mb-8 z-10 relative">
        <div>
          <h3 className="font-headline text-2xl font-bold text-slate-100 tracking-tight">Market Trajectory</h3>
          <p className="text-sm text-slate-400 mt-1 font-body">Algorithmically projected price action bounds</p>
        </div>
        <div className="flex bg-white/5 border border-white/10 p-1 rounded-xl">
          {['1d', '5d', '20d'].map((h) => (
            <button 
              key={h}
              onClick={() => onHorizonChange(h)}
              className={`px-4 py-1.5 text-xs font-bold rounded-lg uppercase tracking-wider transition-all ${
                horizon === h ? 'bg-blue-500 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {h}
            </button>
          ))}
        </div>
      </div>

      <div className="h-[350px] w-full relative z-10">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={true} horizontal={false} stroke="#ffffff" opacity={0.05} />
            <XAxis dataKey="date" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} dy={10} />
            <YAxis domain={['auto', 'auto']} stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} orientation="right" tickFormatter={(v) => v.toLocaleString()} dx={10} />
            
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: 'rgba(255, 255, 255, 0.1)', borderRadius: '12px', backdropFilter: 'blur(10px)', color: '#f8fafc' }}
              itemStyle={{ color: '#f8fafc', fontSize: '13px', fontWeight: '600' }}
              labelStyle={{ color: '#94a3b8', fontSize: '11px', marginBottom: '6px' }}
            />
            
            {/* Historical Price */}
            <Area type="monotone" dataKey="price" stroke="#3B82F6" strokeWidth={3} fillOpacity={1} fill="url(#colorPrice)" activeDot={{ r: 6, fill: '#3B82F6', stroke: '#0B0F19', strokeWidth: 2 }} />
            
            {/* Predicted Path */}
            <Area type="monotone" dataKey="predicted" stroke="#10B981" strokeWidth={3} strokeDasharray="6 4" fillOpacity={1} fill="url(#colorPred)" activeDot={{ r: 6, fill: '#10B981' }} />
            
            {data.length > 0 && <ReferenceLine x={data[Math.max(0, data.length - 2)]?.date} stroke="#64748b" strokeDasharray="3 3" opacity={0.5} />}
          </AreaChart>
        </ResponsiveContainer>

        {predictionData && (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="absolute top-4 right-16 bg-white/5 border border-white/10 backdrop-blur-md p-4 rounded-2xl shadow-2xl max-w-[200px]"
          >
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Target ({horizon})</span>
            </div>
            <div className="text-2xl font-headline font-extrabold text-white tabular-nums tracking-tight">
              ₹{predictionData.targetPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </div>
            <div className={`text-sm font-bold tabular-nums mt-1 ${predictionData.expectedReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {predictionData.expectedReturn >= 0 ? '+' : ''}{(predictionData.expectedReturn * 100).toFixed(2)}% Outlook
            </div>
          </motion.div>
        )}
      </div>

    </motion.div>
  );
};

export default TrendChart;
