import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { evalAPI } from '../services/api';
import { getMetricColor } from '../utils/metrics';
import QuestionDetailModal from '../components/QuestionDetailModal';
import SourceMetricsTable from '../components/SourceMetricsTable';
import QuestionsTable from '../components/QuestionsTable';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

export default function EvalResult() {
    const { filename } = useParams();
    const navigate = useNavigate();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [selectedQuestion, setSelectedQuestion] = useState(null);

    useEffect(() => {
        loadResult();
    }, [filename]);

    async function loadResult() {
        try {
            const res = await evalAPI.getHistoryFile(filename);
            setData(res.data);
        } catch (err) {
            console.error('Failed to load result:', err);
        } finally {
            setLoading(false);
        }
    }



    if (loading) {
        return (
            <div className="loading-container">
                <div className="loading-spinner"></div>
                <span>Loading...</span>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">❌</div>
                <div className="empty-state-text">Failed to load result file.</div>
            </div>
        );
    }

    // Prepare chart data
    const chartData = data.by_source
        ? Object.entries(data.by_source).map(([source, m]) => ({
            source,
            'Hit Rate': +(m.hit_rate * 100).toFixed(1),
            'MRR': +(m.mrr * 100).toFixed(1),
            'MAP': +(m.map * 100).toFixed(1),
            'Pass Rate': +(m.generation_pass_rate * 100).toFixed(1),
        }))
        : [];

    return (
        <div>
            <div className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <button className="btn btn-secondary btn-sm" onClick={() => navigate('/history')}>
                        ← Back
                    </button>
                    <div>
                        <h1 className="page-title" style={{ fontFamily: 'monospace', fontSize: '1.3rem' }}>{filename}</h1>
                        <p className="page-subtitle">Evaluation Result Details</p>
                    </div>
                </div>
            </div>

            {/* Overall Metrics */}
            {data.overall && (
                <div className="section">
                    <h2 className="section-title">📊 Overall Metrics</h2>
                    <div className="card-grid">
                        {[
                            ['Hit Rate', data.overall.hit_rate],
                            ['Partial Hit Rate', data.overall.partial_hit_rate],
                            ['MRR', data.overall.mrr],
                            ['MAP', data.overall.map],
                            ['Pass Rate', data.overall.generation_pass_rate],
                        ].map(([label, value]) => (
                            <div className="stat-card" key={label}>
                                <div className={`stat-value ${getMetricColor(value)}`} style={{
                                    background: 'none',
                                    WebkitTextFillColor: 'unset',
                                    color: value >= 0.7 ? 'var(--accent-green)' : value >= 0.5 ? 'var(--accent-orange)' : 'var(--accent-red)',
                                }}>
                                    {(value * 100).toFixed(1)}%
                                </div>
                                <div className="stat-label">{label}</div>
                            </div>
                        ))}
                        <div className="stat-card">
                            <div className="stat-value">{data.overall.total_questions}</div>
                            <div className="stat-label">Total Questions</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                                {data.overall.avg_total_time?.toFixed(2)}s
                            </div>
                            <div className="stat-label">Avg Time</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Chart: By Source */}
            {chartData.length > 0 && (
                <div className="section">
                    <h2 className="section-title">📈 By Source Dataset</h2>
                    <div className="card" style={{ padding: '24px' }}>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="source" stroke="#a0a0b8" fontSize={12} />
                                <YAxis stroke="#a0a0b8" fontSize={12} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                                <Tooltip
                                    contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                                    labelStyle={{ color: '#e0e0e0' }}
                                    formatter={(value) => `${value}%`}
                                />
                                <Legend />
                                <Bar dataKey="Hit Rate" fill="#4fc3f7" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="MRR" fill="#66bb6a" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="MAP" fill="#ffa726" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="Pass Rate" fill="#ab47bc" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* By Source Table */}
            {data.by_source && (
                <div className="section">
                    <div className="card">
                        <SourceMetricsTable data={data.by_source} labelKey="Source" tagColor="tag-blue" showAvgTime />
                    </div>
                </div>
            )}

            {/* By Question Type */}
            {data.by_question_type && Object.keys(data.by_question_type).length > 0 && (
                <div className="section">
                    <h2 className="section-title">📋 By Question Type</h2>
                    <div className="card">
                        <SourceMetricsTable data={data.by_question_type} labelKey="Type" tagColor="tag-green" />
                    </div>
                </div>
            )}

            {/* Per-Question Details */}
            {data.questions && data.questions.length > 0 && (
                <div className="section">
                    <h2 className="section-title">🔍 Per-Question Details ({data.questions.length})</h2>
                    <div className="card">
                        <QuestionsTable
                            questions={data.questions}
                            onSelectQuestion={setSelectedQuestion}
                        />
                    </div>
                </div>
            )}

            {selectedQuestion && (
                <QuestionDetailModal
                    question={selectedQuestion}
                    onClose={() => setSelectedQuestion(null)}
                />
            )}
        </div>
    );
}
