import React, { useState, useRef, useEffect } from 'react';
import { queryAPI } from '../services/api';
import './Chatbot.css';

const MOCK_SUGGESTIONS = [
  { icon: '📚', text: 'Who has written more than 100 books, Yaşar Kemal or Avram Noam Chomsky?' },
  { icon: '🏀', text: "What contest did Drew Barry's younger brother win in 1996?" },
  { icon: '🍸', text: 'Are both Smoking Bishop and Caipirinha beverages?' }
];

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      sender: 'bot',
      text: '您好！我是 BEARS 智慧助理 🤖 \n 支援 HotpotQA / 2Wiki 等資料集。\n您可以點擊下方按鈕測試複雜題型，或是直接輸入您的問題！',
      isWelcome: true
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    // Add user message
    const userMsg = { id: Date.now().toString(), sender: 'user', text };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsTyping(true);

    try {
      // Call actual backend
      // Using orchestrator default (if agent is null, it routes automatically in backend)
      const res = await queryAPI.query(text);
      const data = res.data;

      const botMsg = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: data.answer || '系統未回傳文字答案。',
        sources: data.retrieved_doc_ids || [],
        reasoning: `[Router / Orchestrator] 決定使用 Agent: ${data.agent_used}`
      };
      
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error('API Error:', error);
      const errorMsg = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: '❌ 抱歉，後端系統處理時發生錯誤。請確認您的 FastAPI 後端伺服器 (8000 port) 是否已啟動。',
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputValue);
    }
  };

  return (
    <div className="section">
      <div className="page-header">
        <h1 className="page-title">智慧助理</h1>
        <p className="page-subtitle">真實串接 BEARS 後端 API，動態調度檢索路徑</p>
      </div>

      <div className="chatbot-container">
        {/* Header */}
        <div className="chatbot-header">
          <div className="chatbot-title-area">
            <div className="chatbot-icon">🤖</div>
            <div>
              <div className="chatbot-title">BEARS Agentic Search (Live)</div>
              <div className="chatbot-subtitle">
                <div className="status-dot"></div>
                已連線至 FastAPI
              </div>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="chatbot-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
              <div className="message-avatar">
                {msg.sender === 'user' ? '🧑‍🏫' : '🐻'}
              </div>
              <div style={{ maxWidth: '100%' }}>
                <div className="message-content">
                  {msg.text.split('\\n').map((line, i) => (
                     <div key={i} style={{ minHeight: '1em' }}>{line}</div>
                  ))}
                </div>
                
                {msg.sender === 'bot' && !msg.isWelcome && (
                  <>
                    {msg.reasoning && (
                      <div className="agent-reasoning">
                        {msg.reasoning.split('\\n').map((line, i) => <div key={i}>{line}</div>)}
                      </div>
                    )}
                    {(msg.sources && msg.sources.length > 0) && (
                      <div className="message-actions">
                        {msg.sources.map((source, i) => (
                          <div key={i} className="source-tag" title={source}>
                            📄 {source.substring(0, 15)}{source.length > 15 ? '...' : ''}
                          </div>
                        ))}
                      </div>
                    )}
                  </>
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

        {/* Suggestions */}
        <div className="chat-suggestions">
          {MOCK_SUGGESTIONS.map((sug, i) => (
            <button 
              key={i} 
              className="suggestion-btn"
              onClick={() => handleSendMessage(sug.text)}
              disabled={isTyping}
            >
              <span>{sug.icon}</span> {sug.text}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="chatbot-input-area">
          <div className="input-wrapper">
            <textarea
              className="chat-input"
              placeholder="請確保後端已啟動，輸入您的問題以進行真實檢索..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows="1"
            />
            <button 
              className="send-btn" 
              onClick={() => handleSendMessage(inputValue)}
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
