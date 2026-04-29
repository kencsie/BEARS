import { useState, useRef, useEffect, useCallback } from 'react';
import './Chatbot.css';

// ── helpers ───────────────────────────────────────────────────────────────────

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function parseContextChunks(contextList) {
  if (!contextList?.length) return [];
  const chunks = [];
  for (const block of contextList) {
    const parts = block.split('\n\n');
    for (const part of parts) {
      const m = part.match(/^\[(\d+)\]\[([^\]]+)\] (.+)/s);
      if (m) chunks.push({ rank: m[1], source: m[2], content: m[3].trim() });
      else if (part.trim()) chunks.push({ rank: '?', source: '', content: part.trim() });
    }
  }
  return chunks;
}

function fmtTime(s) {
  return s != null ? s.toFixed(2) + 's' : '—';
}

function sessionTitle(session) {
  const first = session.messages.find(m => m.sender === 'user');
  return first ? first.text.slice(0, 38) + (first.text.length > 38 ? '…' : '') : '新對話';
}

function sessionDate(session) {
  const d = new Date(session.createdAt);
  return d.toLocaleDateString('zh-TW', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// ── localStorage ──────────────────────────────────────────────────────────────

const STORAGE_KEY = 'bears_chat_sessions';

function loadSessions() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
  catch { return []; }
}

function saveSessions(sessions) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions)); }
  catch { /* quota exceeded — ignore */ }
}

// ── ContextDrawer ─────────────────────────────────────────────────────────────

function ContextDrawer({ context }) {
  const [open, setOpen] = useState(false);
  const chunks = parseContextChunks(context);
  if (!chunks.length) return null;
  return (
    <div style={{ marginTop: '8px' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          background: 'none', border: 'none',
          color: 'var(--accent-blue)', fontSize: '12px',
          cursor: 'pointer', padding: 0,
          display: 'flex', alignItems: 'center', gap: '4px',
        }}
      >
        <span style={{ fontSize: '9px' }}>{open ? '▲' : '▼'}</span>
        📄 參考資料 ({chunks.length} 段)
      </button>
      {open && (
        <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {chunks.map((c, i) => (
            <div key={i} style={{
              background: 'var(--bg-input)',
              border: '1px solid var(--border-color)',
              borderLeft: '3px solid var(--accent-blue)',
              borderRadius: 'var(--radius-sm)',
              padding: '8px 12px',
            }}>
              <div style={{ fontSize: '10px', color: 'var(--accent-cyan)', marginBottom: '4px', fontWeight: 600 }}>
                [{c.rank}]{c.source ? ` · ${c.source}` : ''}
              </div>
              <div style={{
                fontSize: '12px', color: 'var(--text-primary)',
                lineHeight: 1.65, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              }}>
                {c.content}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── BotMessage ────────────────────────────────────────────────────────────────

function BotMessage({ msg }) {
  return (
    <div className="message-wrapper bot">
      <div className="message-avatar">🐻</div>
      <div style={{ maxWidth: '100%', minWidth: 0, flex: 1 }}>
        <div className="message-content">
          {msg.text.split('\n').map((line, i) => (
            <div key={i} style={{ minHeight: '1em' }}>{line || ' '}</div>
          ))}
        </div>

        {msg.meta && (
          <div style={{
            fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px',
            display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center',
          }}>
            <span>⏱ 檢索 {fmtTime(msg.meta.retrieval_time)}</span>
            <span>✍️ 生成 {fmtTime(msg.meta.generation_time)}</span>
            <span>🕐 總計 {fmtTime(msg.meta.total_time)}</span>
            {msg.meta.total_tokens > 0 && <span>🪙 {msg.meta.total_tokens} tokens</span>}
            {msg.tools?.length > 0 && (
              <span style={{ display: 'flex', gap: '3px' }}>
                {msg.tools.map(t => (
                  <span key={t} className="tag tag-orange" style={{ fontSize: '10px' }}>{t}</span>
                ))}
              </span>
            )}
          </div>
        )}

        {msg.context && <ContextDrawer context={msg.context} />}
      </div>
    </div>
  );
}

// ── HistorySidebar ────────────────────────────────────────────────────────────

function HistorySidebar({ sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <div className="chat-sidebar">
      <div className="chat-sidebar-header">
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px' }}>
          對話記錄
        </span>
        <button className="new-chat-btn" onClick={onNew}>+ 新對話</button>
      </div>
      <div className="chat-sidebar-list">
        {sessions.length === 0 && (
          <div style={{ padding: '24px 14px', fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center' }}>
            尚無對話記錄
          </div>
        )}
        {sessions.map(s => (
          <div
            key={s.id}
            className={`chat-session-item${s.id === activeId ? ' active' : ''}`}
            onClick={() => onSelect(s.id)}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="chat-session-title">{sessionTitle(s)}</div>
              <div className="chat-session-date">{sessionDate(s)}</div>
            </div>
            <button
              className="session-delete-btn"
              onClick={e => { e.stopPropagation(); onDelete(s.id); }}
              title="刪除此對話"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Chatbot ───────────────────────────────────────────────────────────────────

const WELCOME_MSG = {
  id: 'welcome',
  sender: 'bot',
  text: '您好！我是 BEARS 智慧助理。\n請輸入您的問題，系統將自動依問題類型選擇檢索策略，給予最合適的回答。',
};

const Chatbot = () => {
  const [sessions, setSessions] = useState(loadSessions);
  const [activeId, setActiveId] = useState(null);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const activeSession = sessions.find(s => s.id === activeId) ?? null;
  const messages = activeSession ? activeSession.messages : [WELCOME_MSG];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const mutateSessions = useCallback((updater) => {
    setSessions(prev => {
      const next = typeof updater === 'function' ? updater(prev) : updater;
      saveSessions(next);
      return next;
    });
  }, []);

  const handleNew = useCallback(() => {
    setActiveId(null);
    setInputValue('');
  }, []);

  const handleSelect = useCallback((id) => {
    setActiveId(id);
    setInputValue('');
  }, []);

  const handleDelete = useCallback((id) => {
    mutateSessions(prev => prev.filter(s => s.id !== id));
    setActiveId(prev => (prev === id ? null : prev));
  }, [mutateSessions]);

  const handleSend = async (text) => {
    if (!text.trim() || isTyping) return;
    const question = text.trim();
    setInputValue('');
    setIsTyping(true);

    const userMsg = { id: genId(), sender: 'user', text: question };

    // Create session on first message
    let sid = activeId;
    if (!sid) {
      sid = genId();
      mutateSessions(prev => [
        { id: sid, createdAt: new Date().toISOString(), messages: [] },
        ...prev,
      ]);
      setActiveId(sid);
    }

    mutateSessions(prev =>
      prev.map(s => s.id === sid ? { ...s, messages: [...s.messages, userMsg] } : s)
    );

    try {
      const res = await fetch('/api/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();

      const botMsg = {
        id: genId(),
        sender: 'bot',
        text: d.answer || '系統未回傳答案。',
        context: d.context || [],
        tools: d.tool_used || [],
        meta: {
          retrieval_time: d.retrieval_time,
          generation_time: d.generation_time,
          total_time: d.total_time,
          total_tokens: d.total_tokens,
        },
      };

      mutateSessions(prev =>
        prev.map(s => s.id === sid ? { ...s, messages: [...s.messages, botMsg] } : s)
      );
    } catch (err) {
      mutateSessions(prev =>
        prev.map(s => s.id === sid ? {
          ...s,
          messages: [...s.messages, {
            id: genId(),
            sender: 'bot',
            text: `❌ 連線失敗：${err.message}。請確認後端已啟動（port 8005）。`,
          }],
        } : s)
      );
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(inputValue);
    }
  };

  return (
    <div className="chatbot-page">
      <HistorySidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={handleSelect}
        onNew={handleNew}
        onDelete={handleDelete}
      />

      <div className="chatbot-main">
        {/* Header */}
        <div className="chatbot-header">
          <div className="chatbot-title-area">
            <div className="chatbot-icon">🐻</div>
            <div>
              <div className="chatbot-title">BEARS 智慧助理</div>
              <div className="chatbot-subtitle">
                <div className="status-dot" />
                Agentic RAG — 多引擎並行檢索 × Cross-Encoder 精排
              </div>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="chatbot-messages">
          {messages.map(msg =>
            msg.sender === 'user' ? (
              <div key={msg.id} className="message-wrapper user">
                <div className="message-avatar">👤</div>
                <div className="message-content">{msg.text}</div>
              </div>
            ) : (
              <BotMessage key={msg.id} msg={msg} />
            )
          )}

          {isTyping && (
            <div className="message-wrapper bot">
              <div className="message-avatar">🐻</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="chatbot-input-area">
          <div className="input-wrapper">
            <textarea
              className="chat-input"
              placeholder="輸入問題，系統將自動判斷問題類型並給予最適回答… (Enter 送出，Shift+Enter 換行)"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={isTyping}
            />
            <button
              className="send-btn"
              onClick={() => handleSend(inputValue)}
              disabled={!inputValue.trim() || isTyping}
            >
              ➤
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
