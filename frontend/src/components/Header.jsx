import React from 'react';
import { Search, Bell } from 'lucide-react';

const Header = () => {
  return (
    <header className="flex justify-between items-center w-full px-8 h-20 sticky top-0 z-[50] bg-[#070A11]/60 backdrop-blur-3xl border-b border-white/5 font-headline text-sm shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
      {/* Subtle top light reflection */}
      <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-blue-500/10 to-transparent" />
      
      <div className="flex items-center gap-6 relative z-10 w-1/3">
        <div className="relative group w-full max-w-sm">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 w-4 h-4 group-focus-within:text-blue-400 transition-colors drop-shadow-sm" />
          <input 
            type="text" 
            placeholder="Command & Search (⌘K)" 
            className="w-full bg-white/[0.03] border border-white/10 rounded-xl pl-10 pr-4 py-2.5 text-xs focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all text-slate-200 placeholder:text-slate-500 focus:outline-none focus:bg-white/[0.06] shadow-[inset_0_2px_4px_rgba(0,0,0,0.3)] hover:border-white/20"
          />
        </div>
      </div>

      <div className="flex items-center gap-5 relative z-10">
        <button className="relative p-2.5 text-slate-400 hover:text-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 hover:bg-blue-500/10 transition-all rounded-xl border border-transparent hover:border-blue-500/30">
          <Bell className="w-5 h-5 drop-shadow-md" />
          <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,1)] border-2 border-[#070A11]" />
        </button>
        <div className="w-9 h-9 rounded-xl overflow-hidden shadow-lg flex items-center justify-center bg-gradient-to-tr from-blue-600 to-indigo-500 text-white font-bold text-xs cursor-pointer hover:shadow-[0_0_20px_rgba(59,130,246,0.6)] transition-all border border-blue-400/40 relative group">
          <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
          <span className="relative z-10 tracking-widest">AP</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
