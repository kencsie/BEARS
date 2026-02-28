import { useState, useEffect } from 'react';
import { experimentsAPI } from '../services/api';

export default function Experiments() {
    const [experiments, setExperiments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [editing, setEditing] = useState(null);
    const [creating, setCreating] = useState(false);

    // Form state
    const [formName, setFormName] = useState('');
    const [formConfig, setFormConfig] = useState({
        model: 'gpt-4o-mini',
        temperature: 0.0,
        top_k: 5,
        rerank_alpha: 0.7,
        rerank_beta: 0.3,
        agent: 'hybrid',
    });

    useEffect(() => {
        loadExperiments();
    }, []);

    async function loadExperiments() {
        try {
            const res = await experimentsAPI.list();
            setExperiments(res.data.experiments);
        } catch (err) {
            console.error('Failed to load experiments:', err);
        } finally {
            setLoading(false);
        }
    }

    async function handleSave() {
        try {
            if (creating) {
                await experimentsAPI.create(formName, formConfig);
            } else if (editing) {
                await experimentsAPI.update(editing, formConfig);
            }
            setEditing(null);
            setCreating(false);
            loadExperiments();
        } catch (err) {
            alert('Error: ' + (err.response?.data?.detail || err.message));
        }
    }

    async function handleDelete(name) {
        if (!confirm(`Delete experiment "${name}"?`)) return;
        try {
            await experimentsAPI.delete(name);
            loadExperiments();
        } catch (err) {
            alert('Error: ' + (err.response?.data?.detail || err.message));
        }
    }

    function startEdit(exp) {
        setEditing(exp.name);
        setCreating(false);
        setFormConfig({ ...exp.config });
    }

    function startCreate() {
        setCreating(true);
        setEditing(null);
        setFormName('');
        setFormConfig({
            model: 'gpt-4o-mini',
            temperature: 0.0,
            top_k: 5,
            rerank_alpha: 0.7,
            rerank_beta: 0.3,
            agent: 'hybrid',
        });
    }

    function cancelForm() {
        setEditing(null);
        setCreating(false);
    }

    if (loading) {
        return (
            <div className="loading-container">
                <div className="loading-spinner"></div>
                <span>Loading...</span>
            </div>
        );
    }

    return (
        <div>
            <div className="page-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <h1 className="page-title">Experiments</h1>
                        <p className="page-subtitle">管理實驗參數 YAML 檔案</p>
                    </div>
                    <button className="btn btn-primary" onClick={startCreate}>
                        + New Experiment
                    </button>
                </div>
            </div>

            {/* Create / Edit Form */}
            {(creating || editing) && (
                <div className="eval-config-panel" style={{ marginBottom: '24px' }}>
                    <div className="panel-title">
                        {creating ? '建立新實驗參數' : `編輯: ${editing}`}
                    </div>

                    {creating && (
                        <div className="form-group">
                            <label className="form-label">Name</label>
                            <input
                                className="form-input"
                                type="text"
                                placeholder="e.g. exp_topk10"
                                value={formName}
                                onChange={(e) => setFormName(e.target.value)}
                            />
                        </div>
                    )}

                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Model</label>
                            <input className="form-input" value={formConfig.model}
                                onChange={(e) => setFormConfig({ ...formConfig, model: e.target.value })} />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Temperature</label>
                            <input className="form-input" type="number" step="0.1" min="0" max="2"
                                value={formConfig.temperature}
                                onChange={(e) => setFormConfig({ ...formConfig, temperature: parseFloat(e.target.value) })} />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Top K</label>
                            <input className="form-input" type="number" min="1" max="20"
                                value={formConfig.top_k}
                                onChange={(e) => setFormConfig({ ...formConfig, top_k: parseInt(e.target.value) })} />
                        </div>
                    </div>

                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Rerank Alpha</label>
                            <input className="form-input" type="number" step="0.1" min="0" max="1"
                                value={formConfig.rerank_alpha}
                                onChange={(e) => setFormConfig({ ...formConfig, rerank_alpha: parseFloat(e.target.value) })} />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Rerank Beta</label>
                            <input className="form-input" type="number" step="0.1" min="0" max="1"
                                value={formConfig.rerank_beta}
                                onChange={(e) => setFormConfig({ ...formConfig, rerank_beta: parseFloat(e.target.value) })} />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Default Agent</label>
                            <select className="form-select" value={formConfig.agent}
                                onChange={(e) => setFormConfig({ ...formConfig, agent: e.target.value })}>
                                <option value="hybrid">hybrid</option>
                                <option value="kg">kg</option>
                                <option value="agentic">agentic</option>
                            </select>
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button className="btn btn-primary" onClick={handleSave}>
                            {creating ? 'Create' : 'Save'}
                        </button>
                        <button className="btn btn-secondary" onClick={cancelForm}>Cancel</button>
                    </div>
                </div>
            )}

            {/* Experiment List */}
            {experiments.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-state-icon">⚗️</div>
                    <div className="empty-state-text">尚無實驗參數。點擊上方按鈕建立。</div>
                </div>
            ) : (
                <div>
                    {experiments.map((exp) => (
                        <div className="card" key={exp.name} style={{ marginBottom: '12px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <div>
                                    <div style={{ fontWeight: 600, fontSize: '1rem', marginBottom: '8px' }}>
                                        {exp.name}
                                        <span style={{ marginLeft: '8px', fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                            {exp.filename}
                                        </span>
                                    </div>
                                    {exp.config && (
                                        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                                            {Object.entries(exp.config).map(([key, value]) => (
                                                <span key={key} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                    <span style={{ color: 'var(--text-muted)' }}>{key}:</span>{' '}
                                                    <span style={{ fontFamily: 'monospace', color: 'var(--accent-cyan)' }}>{String(value)}</span>
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                                <div style={{ display: 'flex', gap: '6px' }}>
                                    <button className="btn btn-secondary btn-sm" onClick={() => startEdit(exp)}>Edit</button>
                                    {exp.name !== 'default' && (
                                        <button className="btn btn-secondary btn-sm" style={{ color: 'var(--accent-red)' }}
                                            onClick={() => handleDelete(exp.name)}>Delete</button>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
