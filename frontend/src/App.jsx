import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import './App.css';

import Dashboard from './pages/Dashboard';
import History from './pages/History';
import EvalResult from './pages/EvalResult';
import Experiments from './pages/Experiments';

function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-logo">BEARS</div>
          <div className="sidebar-subtitle">RAG Evaluation Platform</div>
          <nav className="sidebar-nav">
            <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">📊</span> Dashboard
            </NavLink>
            <NavLink to="/history" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">📁</span> History
            </NavLink>
            <NavLink to="/experiments" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">⚗️</span> Experiments
            </NavLink>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="/results/:filename" element={<EvalResult />} />
            <Route path="/experiments" element={<Experiments />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
