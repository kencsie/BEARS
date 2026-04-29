import { useState, useRef, useEffect } from 'react';
import { generateAPI, retrieveAPI } from '../services/api';
import './Chatbot.css';

const SUGGESTIONS = [
  { icon: '📚', text: '可用的教學資源有哪些？' },
  { icon: '🎯', text: '幫我針對這個主題進行備課' },
  { icon: '📝', text: '幫我設計此課程的課程評量' },
];

const MODE_OPTIONS = [
  { value: 'retrieve', label: '🔍 檢索模式', desc: '回傳 Q/A/C 標準格式' },
  { value: 'generate', label: '✨ 生成模式', desc: '教師備課 AI 助理' },
];

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      sender: 'bot',
      text: '您好！我是 BEARS 智慧助理 🐻\n選擇下方模式後輸入問題，或點擊建議按鈕快速開始。',
      isWelcome: true,
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [mode, setMode] = useState('generate');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = async (text) => {
    if (!text.trim() || isTyping) return;

    setMessages(prev => [...prev, { id: Date.now().toString(), sender: 'user', text }]);
    setInputValue('');
    setIsTyping(true);

    try {
      let botMsg;

      if (mode === 'retrieve') {
        const res = await retrieveAPI.retrieve(text);
        const d = res.data;
        botMsg = {
          id: (Date.now() + 1).toString(),
          sender: 'bot',
          text: d.answer || '系統未回傳答案。',
          context: d.context || [],
          meta: {
            retrieval_time: d.retrieval_time,
            generation_time: d.generation_time,
            total_time: d.total_time,
            total_tokens: d.total_tokens,
          },
          mode: 'retrieve',
        };
      } else {
        const res = await generateAPI.generate(text, 'educational');
        const d = res.data;
        botMsg = {
          id: (Date.now() + 1).toString(),
          sender: 'bot',
          text: d.generated_content || '系統未回傳內容。',
          context: d.context || [],
          meta: {
            retrieval_time: d.retrieval_time,
            generation_time: d.generation_time,
            total_time: d.total_time,
            total_tokens: d.total_tokens,
          },
          mode: 'generate',
        };
      }

      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.error('API Error:', err);
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: '❌ 後端連線失敗，請確認 FastAPI 伺服器已啟動（port 8005）。',
      }]);
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
    <div className="section">
      <div className="page-header">
        <h1 className="page-title">🐻 BEARS 智慧助理</h1>
        <p className="page-subtitle">Agentic RAG — 多引擎並行檢索 × Cross-Encoder 精排</p>
      </div>

      {/* Mode selector */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
        {MODE_OPTIONS.map(opt => (
          <button
            key={opt.value}
            onClick={() => setMode(opt.value)}
            style={{
              padding: '6px 14px',
              borderRadius: '20px',
              border: '1px solid',
              borderColor: mode === opt.value ? '#4f8ef7' : '#ccc',
              background: mode === opt.value ? '#eef3ff' : '#fff',
              color: mode === opt.value ? '#2563eb' : '#555',
              cursor: 'pointer',
              fontWeight: mode === opt.value ? 600 : 400,
              fontSize: '13px',
            }}
          >
            {opt.label}
            <span style={{ marginLeft: '6px', fontSize: '11px', color: '#888' }}>{opt.desc}</span>
          </button>
        ))}
      </div>

      <div className="chatbot-container">
        <div className="chatbot-header">
          <div className="chatbot-title-area">
            <div className="chatbot-icon">🐻</div>
            <div>
              <div className="chatbot-title">BEARS Agentic Search</div>
              <div className="chatbot-subtitle">
                <div className="status-dot"></div>
                {mode === 'generate' ? '生成模式 — 教師備課助理' : '檢索模式 — 標準 Q/A/C 輸出'}
              </div>
            </div>
          </div>
        </div>

        <div className="chatbot-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
              <div className="message-avatar">{msg.sender === 'user' ? '🧑‍🏫' : '🐻'}</div>
              <div style={{ maxWidth: '100%' }}>
                <div className="message-content">
                  {msg.text.split('\n').map((line, i) => (
                    <div key={i} style={{ minHeight: '1em' }}>{line}</div>
                  ))}
                </div>

                {msg.meta && (
                  <div style={{ fontSize: '11px', color: '#888', marginTop: '4px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    <span>⏱ 檢索 {msg.meta.retrieval_time?.toFixed(2)}s</span>
                    <span>✍️ 生成 {msg.meta.generation_time?.toFixed(2)}s</span>
                    <span>🕐 總計 {msg.meta.total_time?.toFixed(2)}s</span>
                    {msg.meta.total_tokens > 0 && <span>🪙 {msg.meta.total_tokens} tokens</span>}
                  </div>
                )}

                {msg.context && msg.context.length > 0 && (
                  <details style={{ marginTop: '6px', fontSize: '12px', color: '#555' }}>
                    <summary style={{ cursor: 'pointer', color: '#4f8ef7' }}>
                      📄 參考資料 ({msg.context.length} 段)
                    </summary>
                    <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      {msg.context.map((chunk, i) => (
                        <div key={i} style={{
                          background: '#f8f9fa',
                          borderLeft: '3px solid #4f8ef7',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          whiteSpace: 'pre-wrap',
                          maxHeight: '120px',
                          overflowY: 'auto',
                        }}>
                          {chunk.substring(0, 300)}{chunk.length > 300 ? '...' : ''}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="message-wrapper bot">
              <div className="message-avatar">🐻</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-suggestions">
          {SUGGESTIONS.map((sug, i) => (
            <button
              key={i}
              className="suggestion-btn"
              onClick={() => handleSend(sug.text)}
              disabled={isTyping}
            >
              <span>{sug.icon}</span> {sug.text}
            </button>
          ))}
        </div>

        <div className="chatbot-input-area">
          <div className="input-wrapper">
            <textarea
              className="chat-input"
              placeholder={mode === 'generate' ? '輸入教學問題，AI 助理將根據知識庫生成備課內容...' : '輸入問題，系統將回傳 Q/A/C 標準格式...'}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows="1"
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
