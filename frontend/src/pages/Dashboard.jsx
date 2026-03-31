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
        const niftyPred = validPreds.find(p => p.ticker === 'SBIN_NS');
        generateTrendChartPath(niftyPred, horizon);

      } catch (e) {
        console.error("Dashboard init error", e);
      } finally {
        setLoading(false);
      }
    };
    fetchDashboardData();
  }, [horizon]);

  const generateTrendChartPath = (niftyPred, hor) => {
    // Generates a mock chart path for the UI using the real prediction endpoint data
    const points = [];
    const basePrice = niftyPred ? niftyPred.current_price : 22100;
    const targetRet = niftyPred ? niftyPred.prediction.predicted_return : 0.01;
    const days = hor === '1d' ? 1 : hor === '5d' ? 5 : 20;
    
    // Historical 30 days
    let currentP = basePrice * 0.95;
    for(let i=-30; i<=0; i++) {
      points.push({
        date: new Date(Date.now() + i * 86400000).toLocaleDateString(),
        price: currentP,
        predicted: null
      });
      currentP += (Math.random() - 0.45) * 100; // random walk upward bias
    }
    
    points[points.length-1].price = basePrice;
    points[points.length-1].predicted = basePrice;

    // Predicted path
    const targetPrice = basePrice * (1 + targetRet);
    let predP = basePrice;
    const step = (targetPrice - basePrice) / days;
    
    for(let i=1; i<=days; i++) {
        predP += step + (Math.random() - 0.5) * 20;
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
      </div>

      {/* SHAP Explanation Modal */}
      {selectedStock && (
        <SHAPModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </React.Fragment>
  );
};

export default Dashboard;
