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
  const [chartTicker, setChartTicker] = useState('');

  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      try {
        const regimeData = await api.getRegime();
        setRegime(regimeData);

        const targetTickers = ["RELIANCE_NS", "HDFCBANK_NS", "SBIN_NS", "TCS_NS", "ICICIBANK_NS"];
        const predsResponses = await Promise.allSettled(
          targetTickers.map(t => api.getPrediction(t, horizon))
        );
        const validPreds = predsResponses
          .filter(r => r.status === 'fulfilled')
          .map(r => r.value);
        
        setPredictions(validPreds);
        
        // Default to existing selection, or first prediction
        const defaultTicker = chartTicker || (validPreds[0]?.ticker?.replace('_NS', '') || '');
        const chartPred = validPreds.find(p => p.ticker.replace('_NS', '') === defaultTicker) || validPreds[0];
        if (chartPred) setChartTicker(chartPred.ticker.replace('_NS', ''));
        await generateTrendChartPath(chartPred, horizon);
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
       points.push({ 
           date: new Date().toLocaleDateString(), 
           price: basePrice, 
           predicted: basePrice 
       });
    }

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

  const handleChartTickerChange = async (tickerName) => {
    setChartTicker(tickerName);
    const fullTicker = tickerName + '_NS';
    const pred = predictions.find(p => p.ticker === fullTicker);
    if (pred) {
      await generateTrendChartPath(pred, horizon);
    }
  };

  return (
    <React.Fragment>
      <div className="space-y-6 pb-20">
        
        {/* Top Section: Regime + Forecasts */}
        <section className="grid grid-cols-1 xl:grid-cols-12 gap-4">
          <HeroForecast 
            regime={regime?.regime || 'loading...'} 
            regimeSince={regime?.since}
            volatility="Normal"
          />
          <ForecastBentoGrid 
            forecasts={[...predictions].sort((a,b) => (b.prediction?.probability || 0) - (a.prediction?.probability || 0)).slice(0, 3)} 
            onCardClick={(stock) => setSelectedStock(stock)}
          />
        </section>

        {/* Chart + Movers */}
        <section className="grid grid-cols-1 xl:grid-cols-12 gap-4">
          <TrendChart 
            data={chartData} 
            horizon={horizon} 
            onHorizonChange={setHorizon}
            predictionData={predictionData}
            ticker={chartTicker}
            availableStocks={predictions.map(p => p.ticker.replace('_NS', ''))}
            onTickerChange={handleChartTickerChange}
          />
          
          <div className="xl:col-span-4 space-y-4">
            <MoversList predictions={predictions} title="Top Gainers" isGainer={true} />
            <MoversList predictions={predictions} title="Top Losers" isGainer={false} />
            
            {/* Analysis Brief */}
            <div className="p-5 bg-primary/5 rounded-2xl text-on-surface">
              <div className="flex items-center gap-2 mb-3">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                <span className="text-[10px] font-semibold uppercase tracking-widest text-primary">Analysis Brief</span>
              </div>
              <p className="text-sm leading-relaxed text-on-surface-variant">
                Market depth confirms <strong className="text-on-surface capitalize">{regime ? regime.regime : '...'}</strong> conditions. 
                Models suggest <strong className="text-gain">{predictions.filter(p => (p.prediction?.predicted_return || 0) >= 0).length}</strong> of {predictions.length} sampled stocks will experience upward momentum in the next {horizon}.
              </p>
            </div>
          </div>
        </section>

        {/* About Section */}
        <section className="mt-4">
          <div className="relative overflow-hidden rounded-2xl bg-surface-container p-10 md:p-12">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-10 items-center">
              <div className="md:col-span-3 space-y-5">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-primary">Open Source</p>
                <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-on-surface">
                  About <span className="text-primary">AlphaStock</span>
                </h2>
                <p className="text-on-surface-variant leading-relaxed text-base">
                  AlphaStock is an advanced, AI-driven ecosystem designed to bring institutional-grade predictive analytics to everyday investors. By synthesizing real-time market data, regime detection algorithms, and powerful machine learning models, we expose actionable insights hidden in the market noise.
                </p>
                <p className="text-on-surface-variant/60 leading-relaxed text-sm">
                  <strong className="text-on-surface-variant font-semibold">Why rely on us?</strong> We eliminate behavioral bias by relying purely on robust statistical confidence. From short-term momentum signals to long-term forecasting using XGBoost and deep ensemble models, our platform provides precise conviction ratings.
                </p>
              </div>
              
              <div className="md:col-span-2 flex flex-col items-center md:items-end">
                <div className="bg-surface-container-high rounded-xl p-6 w-full max-w-sm transition-colors hover:bg-surface-bright duration-200">
                  <div className="flex items-center gap-4 mb-5">
                    <div className="p-2.5 bg-surface-container rounded-lg">
                      <svg className="w-6 h-6 text-on-surface" viewBox="0 0 24 24" fill="currentColor">
                        <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.699-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.646 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.376.202 2.394.1 2.646.64.699 1.026 1.592 1.026 2.683 0 3.842-2.337 4.687-4.565 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48A10.001 10.001 0 0022 12c0-5.523-4.477-10-10-10z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-on-surface font-semibold text-base">GitHub</h3>
                      <p className="text-tertiary text-sm font-medium">Open Source</p>
                    </div>
                  </div>
                  <p className="text-on-surface-variant/60 text-sm mb-6 leading-relaxed">
                    Review our models, UI components, and API integration.
                  </p>
                  <a 
                    href="https://github.com/pansariabhijay-source/Stock_Web_App" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="flex items-center justify-center gap-2 w-full px-5 py-2.5 font-medium text-sm text-on-surface bg-surface-container rounded-lg hover:bg-surface-container-highest transition-colors"
                  >
                    View on GitHub
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <div className="mt-6 text-center text-on-surface-variant/40 text-sm">
          Built by <span className="text-primary font-medium">Abhijay Pansari</span>
        </div>
      </div>

      {/* SHAP Modal */}
      {selectedStock && (
        <SHAPModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </React.Fragment>
  );
};

export default Dashboard;
