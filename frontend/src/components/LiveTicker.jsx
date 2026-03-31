import React from 'react';

// Hardcoded for the ticker animation placeholder
const MOCK_TICKERS = [
  { symbol: "NIFTY 50", price: "22,147.20", change: "+1.24%", type: "positive" },
  { symbol: "RELIANCE", price: "2,984.15", change: "+0.85%", type: "positive" },
  { symbol: "HDFCBANK", price: "1,452.90", change: "-0.42%", type: "negative" },
  { symbol: "INFY", price: "1,620.45", change: "+2.15%", type: "positive" },
  { symbol: "TCS", price: "4,102.30", change: "-1.10%", type: "negative" },
  { symbol: "ICICIBANK", price: "1,050.20", change: "+0.45%", type: "positive" },
  { symbol: "SBIN", price: "720.50", change: "+1.12%", type: "positive" },
  { symbol: "ITC", price: "410.75", change: "-0.25%", type: "negative" },
];

const TickerItem = ({ item }) => (
  <div className="flex items-center gap-2">
    <span className="font-headline font-bold text-xs">{item.symbol}</span>
    <span className="tabular-nums font-medium text-xs text-white">{item.price}</span>
    <span className={`tabular-nums text-[10px] font-bold ${item.type === 'positive' ? 'text-secondary' : 'text-tertiary'}`}>
      {item.change}
    </span>
  </div>
);

const LiveTicker = () => {
  return (
    <div className="h-10 bg-surface-container-low border-b border-outline-variant/10 flex items-center overflow-hidden w-full">
      <div className="animate-ticker flex w-max">
        {/* Render 3 sets to ensure a seamless infinite scroll loop */}
        {[1, 2, 3].map((setIndex) => (
          <div key={`set-${setIndex}`} className="flex items-center gap-8 px-4 shrink-0 pointer-events-none">
            {MOCK_TICKERS.map((item, i) => (
              <TickerItem key={`${setIndex}-${i}`} item={item} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default LiveTicker;
