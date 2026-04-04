import React, { useEffect, useState, useRef, memo } from 'react';
import { api } from '../api';
import { TrendingUp, TrendingDown } from 'lucide-react';

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
      const timer = setTimeout(() => setFlashClass(''), 800);
      return () => clearTimeout(timer);
    }
  }, [price]);

  const isUp = pctChange >= 0;

  return (
    <div className={`flex items-center px-6 gap-2.5 text-sm font-body cursor-default h-full ${flashClass}`}>
      <span className="font-semibold text-on-surface tracking-wide text-xs">{ticker}</span>
      <span className="tabular-nums text-on-surface-variant font-medium text-xs">₹{price}</span>
      <span className={`tabular-nums text-xs font-semibold flex items-center gap-0.5 ${
        isUp ? "text-gain" : "text-loss"
      }`}>
        {isUp ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
        {isUp ? "+" : ""}{pctChange}%
      </span>
    </div>
  );
});

export function LiveTickerTape() {
  const [tickerData, setTickerData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchTicker() {
      try {
        const data = await api.getPrices();
        if (data && data.prices) {
          const liveData = Object.keys(data.prices).map(ticker => {
            const cleanTicker = ticker.replace('_NS', '');
            return {
              ticker: cleanTicker,
              price: data.prices[ticker].price.toFixed(2),
              pctChange: data.prices[ticker].pct_change
            };
          });
          setTickerData(liveData);
        }
      } catch (err) {
        console.error("Failed to load ticker data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchTicker();
    const interval = setInterval(fetchTicker, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading || tickerData.length === 0) return (
    <div className="w-full bg-surface-container-low h-10 border-b border-outline-variant/20 animate-pulse" />
  );

  const duplicatedData = [...tickerData, ...tickerData, ...tickerData, ...tickerData];

  return (
    <div className="w-full overflow-hidden bg-surface-container-low border-b border-outline-variant/20 h-10 flex items-center relative z-20">
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
