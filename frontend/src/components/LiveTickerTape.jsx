import React, { useEffect, useState, useRef, memo } from 'react';
import { api } from '../api';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '../lib/utils';

// Memoized item to prevent re-renders when other prices change,
// and to track its own state for flash animations.
const TickerItem = memo(({ ticker, price, pctChange }) => {
  const [flashClass, setFlashClass] = useState('');
  const prevPriceRef = useRef(price);

  useEffect(() => {
    if (prevPriceRef.current !== price) {
      if (parseFloat(price) > parseFloat(prevPriceRef.current)) {
        setFlashClass('flash-green');
      } else if (parseFloat(price) < parseFloat(prevPriceRef.current)) {
        setFlashClass('flash-red');
      }
      prevPriceRef.current = price;
      
      const timer = setTimeout(() => setFlashClass(''), 1000);
      return () => clearTimeout(timer);
    }
  }, [price]);

  const isUp = pctChange >= 0;

  return (
    <div 
      className={cn(
        "flex items-center px-8 gap-3 text-sm font-body cursor-pointer hover:bg-white/5 rounded-md transition-colors h-full",
        flashClass
      )}
    >
      <span className="font-headline font-bold text-slate-200 tracking-wider ">{ticker}</span>
      <span className="tabular-nums text-slate-300 font-bold">₹{price}</span>
      
      <div className={cn(
        "flex items-center gap-1 tabular-nums font-bold px-2 py-0.5 rounded-md",
        isUp ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
      )}>
        {isUp ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
        {isUp ? "+" : ""}{pctChange}%
      </div>
    </div>
  );
});

export function LiveTickerTape() {
  const [tickerData, setTickerData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchTicker() {
      try {
        const data = await api.getModels();
        // Fallback mock data if API fails or is empty
        let baseStocks = data?.stocks || [];
        if (baseStocks.length === 0) {
           baseStocks = [
             { ticker: 'RELIANCE_NS' }, { ticker: 'HDFCBANK_NS' }, { ticker: 'TCS_NS' },
             { ticker: 'ICICIBANK_NS' }, { ticker: 'INFY_NS' }, { ticker: 'SBIN_NS' }
           ];
        }

        const simulatedLive = baseStocks.map((stock) => {
          const price = (Math.random() * 2000 + 100).toFixed(2);
          const pctChange = (Math.random() * 4 - 2).toFixed(2);
          return {
            ticker: stock.ticker.replace('_NS', ''),
            price,
            pctChange: parseFloat(pctChange),
          };
        });
        setTickerData(simulatedLive);
      } catch (err) {
        console.error("Failed to load ticker data, using fallbacks:", err);
        const fallbacks = ['NIFTY50', 'RELIANCE', 'HDFCBANK', 'TCS', 'ICICIBANK', 'INFY', 'SBIN'];
        setTickerData(fallbacks.map(t => ({
           ticker: t,
           price: (Math.random() * 2000 + 100).toFixed(2),
           pctChange: parseFloat((Math.random() * 4 - 2).toFixed(2))
        })));
      } finally {
        setLoading(false);
      }
    }
    fetchTicker();
  }, []);

  // Update prices slightly to look alive
  useEffect(() => {
    if (loading || tickerData.length === 0) return;

    const interval = setInterval(() => {
      setTickerData((prev) => 
        prev.map(item => {
          // 30% chance to update a specific stock's price to create staggered flashing
          if (Math.random() > 0.3) return item;
          
          const change = (Math.random() - 0.5) * 5;
          const newPrice = Math.max(0, parseFloat(item.price) + change).toFixed(2);
          return { ...item, price: newPrice };
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, [loading, tickerData.length]);

  if (loading || tickerData.length === 0) return (
     <div className="w-full bg-[#0B0F19] h-12 border-b border-white/5 animate-pulse" />
  );

  // Duplicate data 4 times to ensure seamless infinite scrolling for ultra-wide monitors
  const duplicatedData = [...tickerData, ...tickerData, ...tickerData, ...tickerData];

  return (
    <div className="w-full overflow-hidden bg-[#070A11] border-b border-white/5 h-12 flex items-center shadow-lg relative z-20">
      <div className="flex w-max animate-ticker h-full hover:[animation-play-state:paused]">
        {duplicatedData.map((item, idx) => (
          <TickerItem 
            key={`${item.ticker}-${idx}`} 
            ticker={item.ticker} 
            price={item.price} 
            pctChange={item.pctChange} 
          />
        ))}
      </div>
    </div>
  );
}
