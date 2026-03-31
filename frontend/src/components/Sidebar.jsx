import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Filter, TrendingUp, LineChart, Settings2, BarChart2 } from 'lucide-react';
import { cn } from '../lib/utils';

const Sidebar = () => {
  const getLinkClasses = ({ isActive }) => {
    const baseClasses = "relative flex items-center gap-4 px-4 py-3 rounded-xl transition-all duration-300 group overflow-hidden font-bold";
    if (isActive) {
      return cn(baseClasses, "text-white bg-blue-500/10 border border-blue-500/20 shadow-[0_0_20px_rgba(59,130,246,0.1)]");
    }
    return cn(baseClasses, "text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent");
  };

  const NavItem = ({ to, icon: Icon, label }) => (
    <NavLink to={to} className={getLinkClasses}>
      {({ isActive }) => (
        <>
          {isActive && (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-3/4 bg-blue-500 rounded-r-md shadow-[0_0_12px_rgba(59,130,246,0.8)]" />
          )}
          {isActive && (
             <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-transparent opacity-50 pointer-events-none" />
          )}
          <Icon size={18} className={cn("relative z-10 transition-colors duration-300", isActive ? "text-blue-400 drop-shadow-[0_0_8px_rgba(59,130,246,0.8)]" : "group-hover:text-blue-400")} />
          <span className="relative z-10 tracking-wide">{label}</span>
        </>
      )}
    </NavLink>
  );

  return (
    <aside className="flex flex-col py-8 px-5 fixed left-0 top-0 border-r border-white/5 bg-[#070A11]/90 backdrop-blur-2xl h-screen w-64 z-[60] font-headline text-sm shadow-[4px_0_24px_rgba(0,0,0,0.5)]">
      <div className="mb-12 px-2 flex items-center gap-3">
        <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30 overflow-hidden">
          <div className="absolute inset-0 bg-white/20 blur-md transform translate-y-full group-hover:translate-y-0 transition-transform" />
          <BarChart2 className="text-white w-5 h-5 relative z-10" />
        </div>
        <div>
          <h1 className="text-xl font-black text-white leading-none tracking-tight">AlphaStock</h1>
          <p className="text-[9px] text-blue-400 font-bold uppercase tracking-[0.2em] mt-1.5 drop-shadow-[0_0_5px_rgba(59,130,246,0.5)]">Terminal Pro</p>
        </div>
      </div>

      <nav className="flex-1 space-y-3">
        <NavItem to="/" icon={LayoutDashboard} label="Command Center" />
        <NavItem to="/screener" icon={Filter} label="Live Screener" />
        <NavItem to="/analysis" icon={TrendingUp} label="Trend Horizons" />
        <NavItem to="/portfolio" icon={LineChart} label="Model Portfolios" />
      </nav>

      <div className="mt-auto pt-6 border-t border-white/5 relative">
        <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-blue-500/20 to-transparent" />
        <NavItem to="/engine" icon={Settings2} label="System Settings" />
      </div>
    </aside>
  );
};

export default Sidebar;
