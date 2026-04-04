import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Filter, TrendingUp, LineChart, Settings2, BarChart3 } from 'lucide-react';

const Sidebar = () => {
  const getLinkClasses = ({ isActive }) => {
    const base = "flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors duration-200";
    if (isActive) {
      return `${base} bg-primary/10 text-primary`;
    }
    return `${base} text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high`;
  };

  return (
    <aside className="flex flex-col py-6 px-4 fixed left-0 top-0 bg-surface-container-low h-screen w-[240px] z-[60] border-r border-outline-variant/20">
      
      {/* Logo */}
      <div className="mb-10 px-2 flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-primary-container flex items-center justify-center">
          <BarChart3 className="text-on-primary w-5 h-5" />
        </div>
        <div>
          <h1 className="text-base font-bold text-on-surface leading-none tracking-tight">AlphaStock</h1>
          <p className="text-[10px] text-on-surface-variant font-medium uppercase tracking-widest mt-1">Analytics</p>
        </div>
      </div>

      {/* Label */}
      <p className="px-4 mb-2 text-[10px] font-semibold uppercase tracking-widest text-on-surface-variant/50">Navigation</p>

      {/* Nav */}
      <nav className="flex-1 space-y-1">
        <NavLink to="/" className={getLinkClasses}>
          <LayoutDashboard size={18} />
          <span>Dashboard</span>
        </NavLink>
        <NavLink to="/screener" className={getLinkClasses}>
          <Filter size={18} />
          <span>Screener</span>
        </NavLink>
        <NavLink to="/analysis" className={getLinkClasses}>
          <TrendingUp size={18} />
          <span>Analysis</span>
        </NavLink>
        <NavLink to="/portfolio" className={getLinkClasses}>
          <LineChart size={18} />
          <span>Portfolio</span>
        </NavLink>
      </nav>

      {/* Footer */}
      <div className="mt-auto pt-4 border-t border-outline-variant/20">
        <NavLink to="/engine" className={getLinkClasses}>
          <Settings2 size={18} />
          <span>Settings</span>
        </NavLink>
      </div>
    </aside>
  );
};

export default Sidebar;
