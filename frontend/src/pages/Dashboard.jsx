import React, { useState, useEffect } from 'react';
import HeroForecast from '../components/HeroForecast';
import ForecastBentoGrid from '../components/ForecastBentoGrid';
import TrendChart from '../components/TrendChart';
import MoversList from '../components/MoversList';
import SHAPModal from '../components/SHAPModal';
import { api } from '../api';

const Dashboard = () => {
  const [horizon, setHorizon] = useState('1d');
  const [regime, setRegime] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedStock, setSelectedStock] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [predictionData, setPredictionData] = useState(null);

  // Fetch all initial dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      try {
        // 1. Get Regime
        const regimeData = await api.getRegime();
        setRegime(regimeData);

        // 2. Get Available Models & run predictions for Top 8 stocks as a sample
        const modelsRes = await api.getModels();
        // Take a small sample to avoid spamming the backend too hard on first load
        const targetTickers = ["RELIANCE_NS", "HDFCBANK_NS", "SBIN_NS", "TCS_NS", "ICICIBANK_NS"];
        
        const predsResponses = await Promise.allSettled(
          targetTickers.map(t => api.getPrediction(t, horizon))
        );

        const validPreds = predsResponses
          .filter(r => r.status === 'fulfilled')
          .map(r => r.value);
        
        setPredictions(validPreds);
        
        // 3. Setup fake chart path for Nifty (^NSEI) based on prediction
        const niftyPred = validPreds.find(p => p.ticker === 'SBIN_NS') || validPreds[0];
        await generateTrendChartPath(niftyPred, horizon);

      } catch (e) {
        console.error("Dashboard init error", e);
      } finally {
        setLoading(false);
      }
    };
    fetchDashboardData();
  }, [horizon]);

  const generateTrendChartPath = async (niftyPred, hor) => {
    let points = [];
    const basePrice = niftyPred ? niftyPred.current_price : 22100;
    const targetRet = niftyPred ? niftyPred.prediction.predicted_return : 0.01;
    const days = hor === '1d' ? 1 : hor === '5d' ? 5 : 20;

    // Fetch Historical 30 days
    if (niftyPred && niftyPred.ticker) {
      try {
        const histData = await api.getHistory(niftyPred.ticker, 30);
        if (histData && histData.history && histData.history.length > 0) {
           points = histData.history.map(item => ({
             date: new Date(item.date).toLocaleDateString(),
             price: item.price,
             predicted: null
           }));
           points[points.length-1].predicted = points[points.length-1].price;
        }
      } catch (err) {
        console.error("Failed history fetch", err);
      }
    }

    if (points.length === 0) {
       // fallback if history fetch fails
       points.push({ 
           date: new Date().toLocaleDateString(), 
           price: basePrice, 
           predicted: basePrice 
       });
    }

    // Predicted path (smooth linear interpolation)
    const targetPrice = basePrice * (1 + targetRet);
    let predP = basePrice;
    const step = (targetPrice - basePrice) / days;
    
    for(let i=1; i<=days; i++) {
        predP += step;
        points.push({
            date: new Date(Date.now() + i * 86400000).toLocaleDateString(),
            price: null,
            predicted: i === days ? targetPrice : predP
        });
    }
    
    setChartData(points);
    if(niftyPred) {
      setPredictionData({
        targetPrice,
        expectedReturn: niftyPred.prediction.predicted_return,
        lower: niftyPred.prediction.confidence_lower,
        upper: niftyPred.prediction.confidence_upper
      });
    }
  };

  return (
    <React.Fragment>
      <div className="p-8 max-w-[1600px] mx-auto space-y-10 pb-24">
        {/* Top Section */}
        <section className="grid grid-cols-1 xl:grid-cols-12 gap-8">
          <HeroForecast 
            regime={regime?.regime || 'loading...'}
            regimeSince={regime?.since}
            volatility="Normal"
          />
          {/* Show Top 3 absolute conviction predictions */}
          <ForecastBentoGrid 
            forecasts={[...predictions].sort((a,b) => (b.prediction?.probability || 0) - (a.prediction?.probability || 0)).slice(0, 3)} 
            onCardClick={(stock) => setSelectedStock(stock)}
          />
        </section>

        {/* Chart & Tables Section */}
        <section className="grid grid-cols-1 xl:grid-cols-12 gap-8">
          <TrendChart 
            data={chartData} 
            horizon={horizon} 
            onHorizonChange={setHorizon}
            predictionData={predictionData} 
          />
          
          <div className="xl:col-span-4 space-y-8">
            <MoversList predictions={predictions} title="Top Gainers" isGainer={true} />
            <MoversList predictions={predictions} title="Top Losers" isGainer={false} />
            
            {/* Intelligence Brief */}
            <div className="p-8 bg-blue-500/10 border border-blue-500/20 rounded-2xl text-slate-200 shadow-2xl relative overflow-hidden backdrop-blur-xl">
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/20 rounded-full blur-3xl -z-10" />
              <div className="flex items-center gap-3 mb-4">
                <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                <span className="text-[11px] font-bold uppercase tracking-widest text-blue-400">Analysis Brief</span>
              </div>
              <p className="text-sm font-body leading-relaxed text-slate-300">
                Market depth confirms <strong className="text-white capitalize">{regime ? regime.regime : '...'}</strong> conditions. 
                Models suggest <strong className="text-emerald-400">{predictions.filter(p => p.prediction?.direction==='UP').length}</strong> of {predictions.length} sampled stocks will experience upward momentum in the next {horizon}.
              </p>
            </div>
          </div>
        </section>

        {/* About Us Section */}
        <section className="mt-8 w-full">
          <div className="relative overflow-hidden rounded-3xl border border-slate-800/60 bg-gradient-to-b from-slate-900/60 to-[#0A0E17]/80 p-10 backdrop-blur-xl md:p-14 shadow-2xl">
            {/* Decorative blur elements */}
            <div className="absolute -top-32 -left-32 h-72 w-72 rounded-full bg-emerald-500/10 blur-[100px]" />
            <div className="absolute -bottom-32 -right-32 h-72 w-72 rounded-full bg-blue-500/10 blur-[100px]" />
            
            <div className="relative z-10 grid grid-cols-1 md:grid-cols-5 gap-12 items-center">
              <div className="md:col-span-3 space-y-6">
                <div className="inline-flex items-center gap-3 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 mb-2">
                  <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                  <span className="text-xs font-bold uppercase tracking-widest text-blue-400">Open Source</span>
                </div>
                <h2 className="text-3xl md:text-5xl font-black tracking-tight text-white">
                  About <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">AlphaStock</span>
                </h2>
                <p className="text-slate-300 font-body leading-relaxed text-lg">
                  AlphaStock is an advanced, AI-driven ecosystem designed to bring institutional-grade predictive analytics to everyday investors. By synthesizing real-time market data, regime detection algorithms, and powerful machine learning models, we expose actionable insights hidden in the market noise.
                </p>
                <p className="text-slate-400 font-body leading-relaxed text-base">
                  <strong className="text-slate-200 font-semibold">Why rely on us?</strong> We eliminate behavioral bias by relying purely on robust statistical confidence. From short-term momentum signals to long-term forecasting using XGBoost and deep ensemble models, our platform provides precise conviction ratings to guide and optimize your portfolio strategies.
                </p>
              </div>
              
              <div className="md:col-span-2 flex flex-col items-center md:items-end">
                <div className="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-8 backdrop-blur-md w-full max-w-sm transition-transform hover:-translate-y-1 duration-300 shadow-xl relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  <div className="relative z-10">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-slate-900/80 rounded-xl border border-slate-700">
                        <svg 
                          className="w-8 h-8 text-white" 
                          viewBox="0 0 24 24" 
                          fill="currentColor"
                        >
                          <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.699-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.646 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.376.202 2.394.1 2.646.64.699 1.026 1.592 1.026 2.683 0 3.842-2.337 4.687-4.565 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48A10.001 10.001 0 0022 12c0-5.523-4.477-10-10-10z" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="text-white font-bold text-xl">GitHub Repo</h3>
                        <p className="text-emerald-400 text-sm font-medium">100% Open Source</p>
                      </div>
                    </div>
                    <p className="text-slate-400 text-sm mb-8 leading-relaxed">
                      Review our models, UI components, and API integration. Star the repository to show your support!
                    </p>
                    <a 
                      href="https://github.com/pansariabhijay-source/Stock_Web_App" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="group/btn relative inline-flex items-center justify-center w-full gap-2 px-6 py-3.5 font-semibold text-white bg-slate-900 border border-slate-600/50 rounded-xl hover:bg-slate-800 transition-all hover:border-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-[#0A0E17]"
                    >
                      <span className="relative z-10 tracking-wide">View on GitHub</span>
                      <svg 
                        className="w-4 h-4 transform group-hover/btn:translate-x-1.5 transition-transform relative z-10" 
                        fill="none" 
                        viewBox="0 0 24 24" 
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                      </svg>
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* SHAP Explanation Modal */}
      {selectedStock && (
        <SHAPModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </React.Fragment>
  );
};

export default Dashboard;
