import React from 'react';
import { motion } from 'framer-motion';
import { Github, TrendingUp, Zap, Shield, Cpu, Code2, LineChart, Server } from 'lucide-react';

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

const FeatureCard = ({ icon: Icon, title, description, color }) => (
  <motion.div 
    variants={itemVariants}
    whileHover={{ y: -5, scale: 1.02 }}
    className="relative group bg-[#0A0E17]/80 border border-white/10 rounded-2xl p-6 overflow-hidden hover:border-white/30 transition-all shadow-xl"
  >
    <div className={`absolute inset-0 bg-gradient-to-br ${color} opacity-0 group-hover:opacity-10 transition-opacity duration-500`} />
    
    <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-6 bg-white/5 border border-white/10 text-white`}>
      <Icon className="w-6 h-6" />
    </div>
    
    <h3 className="text-xl font-headline font-bold text-slate-100 mb-3">{title}</h3>
    <p className="text-sm font-body text-slate-400 leading-relaxed">
      {description}
    </p>
  </motion.div>
);

const About = () => {
  return (
    <div className="p-8 max-w-[1600px] mx-auto space-y-10 pb-24">
      {/* Header Section */}
      <motion.section 
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative bg-gradient-to-b from-blue-900/20 to-transparent border-t border-blue-500/20 rounded-3xl p-10 overflow-hidden"
      >
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-[120px] -z-10" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-[120px] -z-10" />
        
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-bold uppercase tracking-widest mb-6">
            <Code2 size={14} /> Open Source Project
          </div>
          
          <h1 className="text-5xl font-black font-headline text-white mb-6 tracking-tight leading-tight">
            Institutional-Grade AI for <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
              Retail Investors.
            </span>
          </h1>
          
          <p className="text-lg text-slate-300 font-body leading-relaxed mb-8 max-w-3xl">
            AlphaStock Terminal Pro bridges the gap between quantitative hedge funds and everyday traders. 
            By leveraging advanced tree-based ensembles (LightGBM, XGBoost) and Hidden Markov Models, 
            we provide transparent, actionable price action bounds and market regime detection for Nifty 50 stocks.
          </p>
          
          <a 
            href="https://github.com/pansariabhijay-source/Stock_Web_App" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-3 px-6 py-3 rounded-xl bg-white text-slate-900 font-bold font-body hover:bg-slate-200 transition-colors shadow-[0_0_20px_rgba(255,255,255,0.3)]"
          >
            <Github size={20} />
            View Repository on GitHub
          </a>
        </div>
      </motion.section>

      {/* Features Grid */}
      <div className="pt-4">
        <div className="flex items-center gap-3 mb-8">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <h2 className="text-2xl font-black font-headline text-white tracking-tight">Core Capabilities</h2>
        </div>

        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          <FeatureCard 
            icon={Cpu}
            title="Ensemble Modeling"
            color="from-blue-500 to-indigo-500"
            description="Parallel XGBoost and LightGBM models trained on over a decade of historical data, computing complex non-linear relationships to predict 1D, 5D, and 20D horizons."
          />
          <FeatureCard 
            icon={LineChart}
            title="Market Regime Detection"
            color="from-rose-500 to-orange-500"
            description="Utilizes Hidden Markov Models (HMM) to constantly classify the macroeconomic environment into Bull, Bear, Sideways, or Crisis scenarios."
          />
          <FeatureCard 
            icon={Zap}
            title="SHAP Explainability"
            color="from-emerald-500 to-teal-500"
            description="Black-box models are dangerous. We utilize SHAP (SHapley Additive exPlanations) to transparently reveal exactly which features influenced a prediction."
          />
          <FeatureCard 
            icon={Server}
            title="High-Speed Execution"
            color="from-purple-500 to-pink-500"
            description="Models are pre-loaded in memory using advanced FastAPI caching for ~50ms inference times, combined with an optimized asynchronous UI."
          />
        </motion.div>
      </div>

    </div>
  );
};

export default About;
