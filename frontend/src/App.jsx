import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import { LiveTickerTape } from './components/LiveTickerTape';

import Dashboard from './pages/Dashboard';
import Screener from './pages/Screener';
import Analysis from './pages/Analysis';
import Portfolio from './pages/Portfolio';
import Engine from './pages/Engine';

const App = () => {
  return (
    <div className="flex h-screen overflow-hidden bg-[#0A0E17] text-slate-200">
      <Sidebar />
      <main className="ml-64 flex-1 h-screen overflow-y-auto w-full flex flex-col">
        {/* The new Amazon-quality premium ticker strip */}
        <LiveTickerTape />
        
        <div className="flex-1 w-full mx-auto relative px-6 py-6 overflow-x-hidden">
          <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/screener" element={<Screener />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/engine" element={<Engine />} />
        </Routes>
        </div>
      </main>
    </div>
  );
};

export default App;
