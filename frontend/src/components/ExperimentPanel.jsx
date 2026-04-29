// Experiment configuration panel — model, temperature, top_k

export const DEFAULT_EXPERIMENT = {
  model: 'gpt-4o-mini',
  temperature: 0.0,
  top_k: 5,
  agent: 'agentic',
};

const MODELS = [
  'gpt-4o-mini',
  'gpt-4o',
  'gpt-4.1-mini',
  'gpt-4.1',
];

const fieldStyle = {
  display: 'flex', alignItems: 'center', gap: '6px',
  fontSize: '12px', color: 'var(--text-secondary)',
};

const inputStyle = {
  padding: '4px 8px',
  background: 'var(--bg-input)',
  border: '1px solid var(--border-color)',
  borderRadius: 'var(--radius-sm)',
  color: 'var(--text-primary)',
  fontSize: '12px',
};

export default function ExperimentPanel({ value, onChange, disabled }) {
  const update = (key, val) => onChange({ ...value, [key]: val });

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-md)',
      padding: '10px 16px',
      display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap',
    }}>
      <span style={{
        fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.8px', whiteSpace: 'nowrap',
      }}>
        ⚙ 實驗參數
      </span>

      <label style={fieldStyle}>
        模型
        <select
          value={value.model}
          onChange={e => update('model', e.target.value)}
          disabled={disabled}
          style={{ ...inputStyle, paddingRight: '24px' }}
        >
          {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </label>

      <label style={fieldStyle}>
        Temperature
        <input
          type="number" min={0} max={1} step={0.1}
          value={value.temperature}
          onChange={e => update('temperature', Math.round(parseFloat(e.target.value) * 10) / 10 || 0)}
          disabled={disabled}
          style={{ ...inputStyle, width: '60px' }}
        />
      </label>

      <label style={fieldStyle}>
        Top-K
        <input
          type="number" min={1} max={20} step={1}
          value={value.top_k}
          onChange={e => update('top_k', Math.max(1, parseInt(e.target.value) || 5))}
          disabled={disabled}
          style={{ ...inputStyle, width: '60px' }}
        />
      </label>

      {disabled && (
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          執行中，參數鎖定
        </span>
      )}
    </div>
  );
}
