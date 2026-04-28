import { BrowserRouter, Link, Route, Routes, useLocation } from 'react-router-dom';
import Chatbot from './pages/Chatbot';
import EvalBatch from './pages/EvalBatch';
import './App.css';

const NAV_ITEMS = [
  { to: '/', label: '💬 智慧助理' },
  { to: '/eval', label: '📊 批次評測' },
];

function NavBar() {
  const { pathname } = useLocation();
  return (
    <nav style={{
      display: 'flex', gap: '4px', padding: '8px 16px',
      borderBottom: '1px solid #e5e7eb', background: '#fff',
    }}>
      {NAV_ITEMS.map(item => (
        <Link
          key={item.to}
          to={item.to}
          style={{
            padding: '6px 16px', borderRadius: '8px', fontSize: '13px', fontWeight: 500,
            textDecoration: 'none',
            background: pathname === item.to ? '#eef3ff' : 'transparent',
            color: pathname === item.to ? '#2563eb' : '#555',
          }}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Chatbot />} />
          <Route path="/eval" element={<EvalBatch />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;
