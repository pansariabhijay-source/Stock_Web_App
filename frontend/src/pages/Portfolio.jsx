import React from 'react';

const Portfolio = () => {
  return (
    <div className="p-8 space-y-8 flex items-center justify-center min-h-[60vh] flex-col text-center">
      <span className="material-symbols-outlined text-6xl text-primary/30 mb-4" style={{fontVariationSettings: "'FILL' 1"}}>account_balance_wallet</span>
      <h2 className="text-3xl font-headline font-bold text-white">Prediction Portfolio</h2>
      <p className="text-on-surface-variant max-w-md">Your active tracking portfolio is empty. Add instruments from the screener to start monitoring their algorithmic forecasts over time.</p>
      
      <button className="mt-6 px-6 py-2.5 bg-gradient-to-br from-primary to-primary-container text-on-primary font-bold rounded-lg text-sm hover:opacity-90 transition-opacity">
        Explore Markets
      </button>
    </div>
  );
};

export default Portfolio;
