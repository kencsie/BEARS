import { useState, useRef, useEffect, useMemo } from 'react';

// ─── constants ────────────────────────────────────────────────────────────────

const DATASET_TAG = {
  drcd:     'tag-blue',
  squad_v2: 'tag-orange',
  ms_marco: 'tag-green',
  hotpotqa: 'tag-purple',
  '2wiki':  'tag-cyan',
};

// ─── helpers ──────────────────────────────────────────────────────────────────

function fmtTime(s) {
  if (s == null) return '—';
  return s.toFixed(1) + 's';
}

function parseDateFromFilename(name) {
  const m = name.match(/eval_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  if (!m) return name.replace('eval_', '').replace('.json', '');
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return `${months[+m[2]-1]} ${m[3]}  ${m[4]}:${m[5]}`;
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

// ─── HistoryPanel ─────────────────────────────────────────────────────────────

function HistoryPanel({ history, activeFile, onSelect, loading }) {
  return (
    <div style={{
      width: '220px', flexShrink: 0,
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-md)',
      display: 'flex', flexDirection: 'column',
      maxHeight: 'calc(100vh - 200px)',
      position: 'sticky', top: '20px',
    }}>
      <div style={{
        padding: '12px 14px',
        borderBottom: '1px solid var(--border-color)',
        fontSize: '11px', fontWeight: 600,
        color: 'var(--text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.8px',
      }}>
        歷史紀錄 {history.length > 0 && `(${history.length})`}
      </div>
      <div style={{ overflowY: 'auto', flex: 1 }}>
        {loading && (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
            載入中…
          </div>
        )}
        {!loading && history.length === 0 && (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
            尚無評測紀錄
          </div>
        )}
        {history.map(h => (
          <button
            key={h.filename}
            onClick={() => onSelect(h)}
            style={{
              width: '100%', textAlign: 'left',
              padding: '10px 14px',
              background: activeFile === h.filename ? 'rgba(79,195,247,0.08)' : 'transparent',
              borderLeft: activeFile === h.filename ? '3px solid var(--accent-blue)' : '3px solid transparent',
              border: 'none', borderRight: 'none', borderTop: 'none',
              borderBottom: '1px solid var(--border-color)',
              cursor: 'pointer',
              transition: 'background var(--transition-fast)',
            }}
            onMouseEnter={e => { if (activeFile !== h.filename) e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; }}
            onMouseLeave={e => { if (activeFile !== h.filename) e.currentTarget.style.background = 'transparent'; }}
          >
            <div style={{
              fontSize: '12px', fontFamily: 'monospace',
              color: activeFile === h.filename ? 'var(--accent-blue)' : 'var(--text-secondary)',
              marginBottom: '3px', fontWeight: activeFile === h.filename ? 600 : 400,
            }}>
              {parseDateFromFilename(h.filename)}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {h.count} 題 · avg {fmtTime(h.avg_time)}
            </div>
            {h.datasets?.length > 0 && (
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '3px', display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                {h.datasets.map(d => (
                  <span key={d} className={`tag ${DATASET_TAG[d] || 'tag-blue'}`} style={{ fontSize: '9px', padding: '1px 5px' }}>{d}</span>
                ))}
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── StatsBar ─────────────────────────────────────────────────────────────────

function StatsBar({ results }) {
  const stats = useMemo(() => {
    if (!results.length) return null;
    const times = results.map(r => r.total_time || 0);
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const maxTime = Math.max(...times);
    const datasets = [...new Set(results.map(r => r.source_dataset).filter(Boolean))];
    const errorCount = results.filter(r => r.error).length;
    return { count: results.length, avgTime, maxTime, datasets, errorCount };
  }, [results]);

  if (!stats) return null;

  const metrics = [
    { label: '總題數',   value: stats.count,            color: 'var(--accent-blue)' },
    { label: 'Avg Time', value: fmtTime(stats.avgTime),  color: 'var(--accent-cyan)' },
    { label: 'Max Time', value: fmtTime(stats.maxTime),  color: 'var(--accent-orange)' },
    { label: '資料集',   value: stats.datasets.length,   color: 'var(--accent-purple)' },
    ...(stats.errorCount > 0
      ? [{ label: '錯誤', value: stats.errorCount, color: 'var(--accent-red)' }]
      : []
    ),
  ];

  return (
    <div style={{ display: 'flex', gap: '10px', marginBottom: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
      {metrics.map(m => (
        <div key={m.label} style={{
          background: 'var(--bg-card)', border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-sm)', padding: '8px 14px',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px',
          minWidth: '72px',
        }}>
          <span style={{ fontSize: '16px', fontWeight: 700, color: m.color, fontVariantNumeric: 'tabular-nums' }}>{m.value}</span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', whiteSpace: 'nowrap' }}>{m.label}</span>
        </div>
      ))}
      {stats.datasets.length > 0 && (
        <div style={{
          background: 'var(--bg-card)', border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-sm)', padding: '8px 14px',
          display: 'flex', flexDirection: 'column', gap: '5px',
        }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>資料集分布</span>
          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
            {stats.datasets.map(d => (
              <span key={d} className={`tag ${DATASET_TAG[d] || 'tag-blue'}`}>{d}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── DetailModal ──────────────────────────────────────────────────────────────

function DetailModal({ row, onClose }) {
  const chunks = parseContextChunks(row.context);

  useEffect(() => {
    document.body.classList.add('modal-open');
    const handler = e => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => {
      document.body.classList.remove('modal-open');
      document.removeEventListener('keydown', handler);
    };
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-content">
        <div className="modal-header">
          <div>
            <div className="modal-title" style={{ fontSize: '1rem' }}>詳細結果</div>
            <div style={{ display: 'flex', gap: '6px', marginTop: '6px', flexWrap: 'wrap' }}>
              {row.source_dataset && <span className={`tag ${DATASET_TAG[row.source_dataset] || 'tag-blue'}`}>{row.source_dataset}</span>}
              {row.question_type  && <span className="tag tag-cyan">{row.question_type}</span>}
              {(row.tool_used || []).map(t => <span key={t} className="tag tag-orange">{t}</span>)}
            </div>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {/* Timing */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' }}>
            {[
              { label: '檢索時間',   value: fmtTime(row.retrieval_time),   color: 'var(--accent-blue)' },
              { label: '生成時間',   value: fmtTime(row.generation_time),  color: 'var(--accent-cyan)' },
              { label: '總計',       value: fmtTime(row.total_time),        color: 'var(--accent-green)' },
            ].map(m => (
              <div key={m.label} className="detail-metric">
                <span style={{ fontSize: '18px', fontWeight: 700, color: m.color }}>{m.value}</span>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{m.label}</span>
              </div>
            ))}
          </div>

          {/* Question */}
          <div className="detail-field">
            <div className="detail-label">問題</div>
            <div className="detail-value">{row.question}</div>
          </div>

          {/* Answer + Gold side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '16px' }}>
            <div className="detail-field" style={{ margin: 0 }}>
              <div className="detail-label">模型答案</div>
              <div className="detail-value detail-model" style={{ minHeight: '60px' }}>{row.answer || '—'}</div>
            </div>
            <div className="detail-field" style={{ margin: 0 }}>
              <div className="detail-label">Gold Answer</div>
              <div className="detail-value detail-gold" style={{ minHeight: '60px' }}>{row.true_answer || '—'}</div>
            </div>
          </div>

          {/* Context chunks */}
          {chunks.length > 0 && (
            <div className="detail-field">
              <div className="detail-label">參考段落 ({chunks.length})</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
                {chunks.map((c, i) => (
                  <div key={i} style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-color)',
                    borderLeft: '3px solid var(--accent-blue)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '10px 14px',
                  }}>
                    <div style={{ fontSize: '10px', color: 'var(--accent-cyan)', marginBottom: '5px', fontWeight: 600 }}>
                      [{c.rank}]{c.source ? ` · ${c.source}` : ''}
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-primary)', lineHeight: 1.65, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {c.content}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── ResultsTable ─────────────────────────────────────────────────────────────

const TH = {
  padding: '10px 12px', textAlign: 'left', fontWeight: 600,
  color: 'var(--text-muted)', fontSize: '11px',
  textTransform: 'uppercase', letterSpacing: '0.5px',
  borderBottom: '2px solid var(--border-color)',
  whiteSpace: 'nowrap', background: 'var(--bg-secondary)',
};
const TD = {
  padding: '9px 12px', verticalAlign: 'top', fontSize: '12px',
  borderBottom: '1px solid var(--border-color)',
};

function ResultsTable({ results, onRowClick }) {
  if (!results.length) return null;
  return (
    <div style={{
      overflowX: 'auto',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-md)',
      background: 'var(--bg-card)',
    }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={TH}>#</th>
            <th style={TH}>問題</th>
            <th style={TH}>答案摘要</th>
            <th style={TH}>Gold Answer</th>
            <th style={TH}>Dataset</th>
            <th style={TH}>Type</th>
            <th style={TH}>Tools</th>
            <th style={TH}>Time</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr
              key={i}
              className="clickable-row"
              onClick={() => onRowClick(r)}
              style={{
                background: r.error
                  ? 'rgba(239,83,80,0.04)'
                  : i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
              }}
            >
              <td style={{ ...TD, color: 'var(--text-muted)', fontWeight: 600, width: '36px' }}>{i + 1}</td>

              <td style={{ ...TD, maxWidth: '200px' }}>
                <span style={{ color: 'var(--text-primary)', lineHeight: 1.5 }} title={r.question}>
                  {r.question.length > 60 ? r.question.slice(0, 60) + '…' : r.question}
                </span>
              </td>

              <td style={{ ...TD, maxWidth: '220px' }}>
                <span
                  style={{ color: r.error ? 'var(--accent-red)' : 'var(--text-secondary)', lineHeight: 1.5 }}
                  title={r.answer}
                >
                  {(r.answer || '').length > 80 ? (r.answer || '').slice(0, 80) + '…' : (r.answer || '—')}
                </span>
              </td>

              <td style={{ ...TD, maxWidth: '140px', color: 'var(--text-muted)' }}>
                <span title={r.true_answer}>
                  {(r.true_answer || '—').length > 40 ? (r.true_answer || '').slice(0, 40) + '…' : (r.true_answer || '—')}
                </span>
              </td>

              <td style={TD}>
                <span className={`tag ${DATASET_TAG[r.source_dataset] || 'tag-blue'}`}>
                  {r.source_dataset || '—'}
                </span>
              </td>

              <td style={{ ...TD, color: 'var(--text-muted)', fontSize: '11px', whiteSpace: 'nowrap' }}>
                {r.question_type || '—'}
              </td>

              <td style={TD}>
                <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                  {(r.tool_used || []).length
                    ? (r.tool_used || []).map(t => <span key={t} className="tag tag-orange" style={{ fontSize: '10px' }}>{t}</span>)
                    : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                </div>
              </td>

              <td style={{ ...TD, fontFamily: 'monospace', color: 'var(--accent-cyan)', whiteSpace: 'nowrap' }}>
                {r.total_time != null ? r.total_time.toFixed(1) + 's' : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

const EvalBatch = () => {
  const [queries, setQueries]           = useState([]);
  const [limit, setLimit]               = useState(10);
  const [isRunning, setIsRunning]       = useState(false);
  const [results, setResults]           = useState([]);
  const [progress, setProgress]         = useState({ current: 0, total: 0 });
  const [outputFile, setOutputFile]     = useState('');
  const [error, setError]               = useState('');
  const [history, setHistory]           = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [activeFile, setActiveFile]     = useState('');
  const [selectedRow, setSelectedRow]   = useState(null);
  const abortRef = useRef(null);

  const fetchHistory = () =>
    fetch('/api/history')
      .then(r => r.json())
      .then(data => { setHistory(data); setHistoryLoading(false); })
      .catch(() => setHistoryLoading(false));

  useEffect(() => { fetchHistory(); }, []);

  const handleFileLoad = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        if (!Array.isArray(data)) throw new Error('Expected a JSON array');
        setQueries(data);
        setLimit(Math.min(10, data.length));
        setResults([]);
        setOutputFile('');
        setError('');
        setActiveFile('');
        setProgress({ current: 0, total: 0 });
      } catch (err) {
        setError(`JSON 解析錯誤：${err.message}`);
      }
    };
    reader.readAsText(file, 'utf-8');
    e.target.value = '';
  };

  const handleHistorySelect = async (h) => {
    try {
      const res = await fetch(`/api/history/${h.filename}`);
      if (!res.ok) throw new Error(`Error ${res.status}`);
      const data = await res.json();
      setResults(data);
      setActiveFile(h.filename);
      setOutputFile(h.filename);
      setProgress({ current: data.length, total: data.length });
      setError('');
      setQueries([]);
    } catch (e) {
      setError(`載入歷史失敗：${e.message}`);
    }
  };

  const handleRun = async () => {
    if (!queries.length || isRunning) return;
    const selected = queries.slice(0, limit);
    setIsRunning(true);
    setResults([]);
    setProgress({ current: 0, total: limit });
    setOutputFile('');
    setError('');
    setActiveFile('');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ queries: selected, limit }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const msg = await response.text();
        throw new Error(`Server error ${response.status}: ${msg}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data._done) {
              setOutputFile(data.output_file);
              fetchHistory();
            } else {
              if (data._progress) setProgress(data._progress);
              setResults(prev => [...prev, data]);
            }
          } catch { /* skip malformed */ }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') setError(String(e));
    } finally {
      setIsRunning(false);
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setIsRunning(false);
  };

  const handleDownload = () => {
    const clean = results.map(({ _progress, ...r }) => r);
    const blob = new Blob([JSON.stringify(clean, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `eval_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pct = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  return (
    <div className="section">
      <div className="page-header">
        <h1 className="page-title">📊 批次評測</h1>
        <p className="page-subtitle">批次上傳 queries.json → 選題數執行 → 點擊查看詳情 → 匯出 JSON</p>
      </div>

      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>

        {/* ─── History Sidebar ─── */}
        <HistoryPanel
          history={history}
          activeFile={activeFile}
          onSelect={handleHistorySelect}
          loading={historyLoading}
        />

        {/* ─── Main Panel ─── */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '14px' }}>

          {/* Controls bar */}
          <div style={{
            display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap',
            background: 'var(--bg-card)', border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-md)', padding: '14px 16px',
          }}>
            <label style={{
              cursor: 'pointer', padding: '6px 14px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-active)',
              color: 'var(--accent-blue)',
              background: 'rgba(79,195,247,0.06)',
              fontSize: '13px', fontWeight: 500,
            }}>
              📂 載入 queries.json
              <input type="file" accept=".json" onChange={handleFileLoad} style={{ display: 'none' }} />
            </label>

            {queries.length > 0 && (
              <>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  已載入 <b style={{ color: 'var(--text-secondary)' }}>{queries.length}</b> 題
                </span>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                  執行題數
                  <input
                    type="number" min={1} max={queries.length} value={limit}
                    onChange={e => setLimit(Math.max(1, Math.min(queries.length, Number(e.target.value))))}
                    disabled={isRunning}
                    style={{
                      width: '58px', padding: '4px 8px',
                      background: 'var(--bg-input)', border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', fontSize: '12px',
                    }}
                  />
                  <span style={{ color: 'var(--text-muted)' }}>/ {queries.length}</span>
                </div>

                {!isRunning ? (
                  <button onClick={handleRun} className="btn btn-primary btn-sm">
                    ▶ 開始執行
                  </button>
                ) : (
                  <button onClick={handleStop} style={{
                    padding: '6px 14px', borderRadius: 'var(--radius-sm)',
                    background: 'rgba(239,83,80,0.12)', color: 'var(--accent-red)',
                    border: '1px solid rgba(239,83,80,0.3)',
                    fontSize: '13px', fontWeight: 500,
                  }}>
                    ⏹ 停止
                  </button>
                )}
              </>
            )}

            {results.length > 0 && (
              <button onClick={handleDownload} style={{
                marginLeft: 'auto', padding: '6px 14px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid rgba(102,187,106,0.35)',
                background: 'rgba(102,187,106,0.08)',
                color: 'var(--accent-green)', fontSize: '13px', fontWeight: 500,
              }}>
                ⬇ 下載 JSON
              </button>
            )}
          </div>

          {/* Progress bar */}
          {(isRunning || (progress.total > 0 && results.length > 0 && !activeFile)) && (
            <div>
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '6px',
              }}>
                <span>
                  {isRunning ? '⏳ 執行中…' : '✓ 完成'}
                  &nbsp;{progress.current} / {progress.total} 題
                </span>
                {outputFile && (
                  <span style={{ color: 'var(--accent-green)', fontSize: '11px' }}>
                    已儲存 {outputFile.split('/').pop()}
                  </span>
                )}
              </div>
              <div className="progress-bar-wrapper">
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${pct}%`,
                    background: isRunning ? 'var(--gradient-blue)' : 'var(--gradient-green)',
                  }}
                />
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{
              color: 'var(--accent-red)',
              background: 'rgba(239,83,80,0.08)',
              border: '1px solid rgba(239,83,80,0.25)',
              borderRadius: 'var(--radius-sm)',
              padding: '10px 14px', fontSize: '13px',
            }}>
              {error}
            </div>
          )}

          {/* Stats */}
          <StatsBar results={results} />

          {/* Results / Empty state */}
          {results.length > 0 ? (
            <ResultsTable results={results} onRowClick={setSelectedRow} />
          ) : (
            <div className="empty-state" style={{ border: '2px dashed var(--border-color)', borderRadius: 'var(--radius-md)' }}>
              <div className="empty-state-icon">📂</div>
              <div className="empty-state-text">
                載入 queries.json 或從左側歷史紀錄選擇一筆紀錄
              </div>
              <div style={{ fontSize: '11px', marginTop: '6px', color: 'var(--text-muted)' }}>
                格式：[&#123; question_id, question, gold_answer, source_dataset, question_type &#125;, …]
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Detail Modal */}
      {selectedRow && (
        <DetailModal row={selectedRow} onClose={() => setSelectedRow(null)} />
      )}
    </div>
  );
};

export default EvalBatch;
