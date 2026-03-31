import React from 'react';
import { motion, useMotionTemplate, useMotionValue } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import { cn } from '../lib/utils';
import { ArrowUpRight, ArrowDownRight, Target } from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const ForecastBentoGrid = ({ forecasts, onCardClick }) => {
  if (!forecasts || forecasts.length === 0) {
    return (
      <div className="col-span-1 lg:col-span-8 flex items-center justify-center p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-xl">
        <div className="text-slate-400 font-body animate-pulse">Computing Predictive Horizons...</div>
      </div>
    );
  }

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="col-span-1 lg:col-span-8 grid grid-cols-1 md:grid-cols-3 gap-6"
    >
      {forecasts.map((f, i) => (
        <ForecastCard key={`${f.ticker}-${i}`} data={f} onClick={() => onCardClick(f.ticker)} />
      ))}
    </motion.div>
  );
};

const ForecastCard = ({ data, onClick }) => {
  const { ticker, prediction, current_price } = data;
  const isUp = prediction.direction === "UP";
  const expectedReturn = (prediction.predicted_return * 100).toFixed(2);
  const targetPrice = (current_price * (1 + prediction.predicted_return)).toFixed(2);
  const confPercent = Math.round(prediction.probability * 100);

  let mouseX = useMotionValue(0);
  let mouseY = useMotionValue(0);

  function handleMouseMove({ currentTarget, clientX, clientY }) {
    let { left, top } = currentTarget.getBoundingClientRect();
    mouseX.set(clientX - left);
    mouseY.set(clientY - top);
  }

  // Fake miniature sparkline data representing predicted trajectory
  const sparkData = [
    { value: current_price * 0.98 },
    { value: current_price * 0.99 },
    { value: current_price },
    { value: current_price + ((targetPrice - current_price) * 0.5) },
    { value: parseFloat(targetPrice) }
  ];

  return (
    <motion.div 
      variants={itemVariants}
      whileHover={{ y: -5, scale: 1.02 }}
      onClick={onClick}
      onMouseMove={handleMouseMove}
      className="group relative cursor-pointer bg-[#0A0E17]/80 border border-white/10 rounded-2xl p-6 overflow-hidden hover:border-white/30 transition-all shadow-[0_8px_30px_rgb(0,0,0,0.5)]"
    >
      {/* Dynamic Cursor Hover Glow */}
      <motion.div
        className="pointer-events-none absolute -inset-px rounded-2xl opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{
          background: useMotionTemplate`
            radial-gradient(
              400px circle at ${mouseX}px ${mouseY}px,
              rgba(${isUp ? '16, 185, 129' : '244, 63, 94'}, 0.15),
              transparent 80%
            )
          `,
        }}
      />

      <div className="flex justify-between items-start mb-6 relative z-10">
        <div>
          <div className="text-xl font-headline font-bold text-slate-100">{ticker}</div>
          <div className="text-xs font-label text-slate-500">Vol {Math.round(prediction.confidence_upper * 10)}%</div>
        </div>
        <div className={cn(
          "px-3 py-1 rounded-full text-xs font-bold font-body flex items-center gap-1",
          isUp ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : 
                 "bg-rose-500/10 text-rose-400 border border-rose-500/20"
        )}>
          {isUp ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
          {expectedReturn}%
        </div>
      </div>

      <div className="mb-4 relative z-10">
        <div className="text-3xl font-body tabular-nums font-semibold text-white tracking-tight">
          ₹{current_price.toLocaleString()}
        </div>
        <div className="text-xs font-label text-slate-500 flex items-center gap-1 mt-1">
          <Target size={12} /> Target: <span className="text-slate-300 font-medium">₹{targetPrice}</span>
        </div>
      </div>

      {/* Mini Sparkline Chart */}
      <div className="h-16 w-full mt-4 opacity-70 group-hover:opacity-100 transition-opacity relative z-10">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={sparkData}>
            <defs>
              <linearGradient id={`grad-${ticker}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={isUp ? "#10B981" : "#F43F5E"} stopOpacity={0.8}/>
                <stop offset="95%" stopColor={isUp ? "#10B981" : "#F43F5E"} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <YAxis domain={['dataMin', 'dataMax']} hide />
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke={isUp ? "#34D399" : "#FB7185"} 
              strokeWidth={3}
              fillOpacity={1} 
              fill={`url(#grad-${ticker})`} 
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 pt-4 border-t border-white/5 flex justify-between items-center relative z-10">
        <span className="text-xs font-label text-slate-400">Confidence</span>
        <div className="w-1/2 bg-white/10 rounded-full h-1.5 overflow-hidden">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: `${confPercent}%` }}
            transition={{ duration: 1, delay: 0.2 }}
            className={cn("h-full", isUp ? "bg-emerald-500" : "bg-rose-500")}
          />
        </div>
        <span className="text-xs font-bold text-slate-200">{confPercent}%</span>
      </div>
    </motion.div>
  );
};

export default ForecastBentoGrid;
