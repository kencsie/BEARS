import { useState, useEffect } from 'react';
import { evalAPI, experimentsAPI } from '../services/api';
import { getMetricColor } from '../utils/metrics';
import QuestionDetailModal from '../components/QuestionDetailModal';
import SourceMetricsTable from '../components/SourceMetricsTable';
import QuestionsTable from '../components/QuestionsTable';

export default function Dashboard() {
    const [agents, setAgents] = useState([]);
    const [stats, setStats] = useState(null);
    const [experiments, setExperiments] = useState([]);
    const [loading, setLoading] = useState(true);

    // Eval config state
    const [selectedAgent, setSelectedAgent] = useState('');
    const [useOrchestrator, setUseOrchestrator] = useState(false);
    const [limit, setLimit] = useState('');
    const [selectedConfig, setSelectedConfig] = useState('');
    const [detailed, setDetailed] = useState(true);
    const [failuresOnly, setFailuresOnly] = useState(false);
    const [outputFilename, setOutputFilename] = useState('');

    // Task tracking
    const [taskId, setTaskId] = useState(null);
    const [taskStatus, setTaskStatus] = useState(null);
    const [results, setResults] = useState(null);
    const [selectedQuestion, setSelectedQuestion] = useState(null);

    useEffect(() => {
        loadInitialData();
    }, []);

    // Poll task status
    useEffect(() => {
        if (!taskId || taskStatus?.status === 'completed' || taskStatus?.status === 'failed') return;

        const interval = setInterval(async () => {
            try {
                const res = await evalAPI.getStatus(taskId);
                setTaskStatus(res.data);

                if (res.data.status === 'completed') {
                    const resultRes = await evalAPI.getResults(taskId);
                    setResults(resultRes.data.results);
                    clearInterval(interval);
                } else if (res.data.status === 'failed') {
                    clearInterval(interval);
                }
            } catch (err) {
                console.error('Status poll error:', err);
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [taskId, taskStatus?.status]);

    async function loadInitialData() {
        try {
            const [agentsRes, statsRes, expRes] = await Promise.all([
                evalAPI.getAgents(),
                evalAPI.getQueryStats(),
                experimentsAPI.list(),
            ]);
            setAgents(agentsRes.data.agents);
            setStats(statsRes.data);
            setExperiments(expRes.data.experiments);

            const firstEnabled = agentsRes.data.agents.find(a => a.enabled);
            if (firstEnabled) setSelectedAgent(firstEnabled.name);
        } catch (err) {
            console.error('Failed to load data:', err);
        } finally {
            setLoading(false);
        }
    }

    async function handleStartEval() {
        setResults(null);
        setTaskStatus(null);

        try {
            const params = {
                agent: useOrchestrator ? null : selectedAgent,
                orchestrator: useOrchestrator,
                limit: limit ? parseInt(limit) : null,
                config_path: selectedConfig || null,
                detailed,
                failures_only: failuresOnly,
                output_filename: outputFilename || null,
            };

            const res = await evalAPI.start(params);
            setTaskId(res.data.task_id);
            setTaskStatus({ status: 'pending', progress: 0, total: 0, message: res.data.message });
        } catch (err) {
            console.error('Failed to start evaluation:', err);
            alert('Failed to start evaluation: ' + (err.response?.data?.detail || err.message));
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

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">Dashboard</h1>
                <p className="page-subtitle">RAG Agent 效能評估平台</p>
            </div>

            {/* Stats Overview */}
            {stats && (
                <div className="section">
                    <h2 className="section-title">📊 題目統計</h2>
                    <div className="card-grid">
                        <div className="stat-card">
                            <div className="stat-value">{stats.total}</div>
                            <div className="stat-label">Total Questions</div>
                        </div>
                        {Object.entries(stats.by_source).map(([source, count]) => (
                            <div className="stat-card" key={source}>
                                <div className="stat-value">{count}</div>
                                <div className="stat-label">{source}</div>
                            </div>
                        ))}
                        {Object.entries(stats.by_question_type).map(([type, count]) => (
                            <div className="stat-card" key={type}>
                                <div className="stat-value">{count}</div>
                                <div className="stat-label">{type}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Agent Cards */}
            <div className="section">
                <h2 className="section-title">🤖 Available Agents</h2>
                <div className="card-grid">
                    {agents.map(agent => (
                        <div className="card agent-card" key={agent.name} data-agent={agent.name}>
                            <div className="agent-name">{agent.name}</div>
                            <div className="agent-module">{agent.module}</div>
                            <span className={`agent-badge ${agent.enabled ? 'enabled' : 'disabled'}`}>
                                {agent.enabled ? 'Enabled' : 'Disabled'}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Evaluation Config */}
            <div className="section">
                <h2 className="section-title">🚀 Start Evaluation</h2>
                <div className="eval-config-panel">
                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Mode</label>
                            <select
                                className="form-select"
                                value={useOrchestrator ? 'orchestrator' : 'agent'}
                                onChange={(e) => setUseOrchestrator(e.target.value === 'orchestrator')}
                            >
                                <option value="agent">Single Agent</option>
                                <option value="orchestrator">Orchestrator (Router → Agent)</option>
                            </select>
                        </div>

                        {!useOrchestrator && (
                            <div className="form-group">
                                <label className="form-label">Agent</label>
                                <select
                                    className="form-select"
                                    value={selectedAgent}
                                    onChange={(e) => setSelectedAgent(e.target.value)}
                                >
                                    {agents.filter(a => a.enabled).map(a => (
                                        <option key={a.name} value={a.name}>{a.name}</option>
                                    ))}
                                </select>
                            </div>
                        )}

                        <div className="form-group">
                            <label className="form-label">Limit (optional)</label>
                            <input
                                className="form-input"
                                type="number"
                                min="1"
                                max="100"
                                placeholder="All questions"
                                value={limit}
                                onChange={(e) => setLimit(e.target.value)}
                            />
                        </div>
                    </div>

                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label">Experiment Config</label>
                            <select
                                className="form-select"
                                value={selectedConfig}
                                onChange={(e) => setSelectedConfig(e.target.value)}
                            >
                                <option value="">Default</option>
                                {experiments.map(exp => (
                                    <option key={exp.name} value={`experiments/${exp.filename}`}>
                                        {exp.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Output Filename (optional)</label>
                            <input
                                className="form-input"
                                type="text"
                                placeholder="Auto-generated"
                                value={outputFilename}
                                onChange={(e) => setOutputFilename(e.target.value)}
                            />
                        </div>
                    </div>

                    <div className="form-group">
                        <div className="checkbox-group">
                            <label className="checkbox-label">
                                <input type="checkbox" checked={detailed} onChange={(e) => setDetailed(e.target.checked)} />
                                Detailed (per-question results)
                            </label>
                            <label className="checkbox-label">
                                <input type="checkbox" checked={failuresOnly} onChange={(e) => setFailuresOnly(e.target.checked)} />
                                Failures only
                            </label>
                        </div>
                    </div>

                    <button
                        className="btn btn-primary"
                        onClick={handleStartEval}
                        disabled={taskStatus?.status === 'running' || taskStatus?.status === 'pending'}
                    >
                        {taskStatus?.status === 'running' ? '⏳ Running...' : '▶ Start Evaluation'}
                    </button>
                </div>
            </div>

            {/* Progress */}
            {taskStatus && (taskStatus.status === 'running' || taskStatus.status === 'pending') && (
                <div className="section">
                    <h2 className="section-title">⏳ Progress</h2>
                    <div className="card">
                        <div className="progress-container">
                            <div className="progress-bar-wrapper">
                                <div
                                    className="progress-bar-fill"
                                    style={{ width: taskStatus.total > 0 ? `${(taskStatus.progress / taskStatus.total) * 100}%` : '0%' }}
                                ></div>
                            </div>
                            <div className="progress-text">
                                <span>{taskStatus.message}</span>
                                <span>{taskStatus.progress} / {taskStatus.total}</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Results */}
            {results && (
                <div className="section">
                    <h2 className="section-title">📈 Results</h2>

                    {/* Overall Metrics */}
                    {results.overall && (
                        <div className="card" style={{ marginBottom: '20px' }}>
                            <h3 style={{ marginBottom: '12px', fontSize: '1rem' }}>Overall Metrics</h3>
                            <table className="metrics-table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {[
                                        ['Total Questions', results.overall.total_questions, false],
                                        ['Hit Rate', results.overall.hit_rate, true],
                                        ['Partial Hit Rate', results.overall.partial_hit_rate, true],
                                        ['MRR', results.overall.mrr, true],
                                        ['MAP', results.overall.map, true],
                                        ['Generation Pass Rate', results.overall.generation_pass_rate, true],
                                        ['Avg Total Time', `${results.overall.avg_total_time?.toFixed(2)}s`, false],
                                    ].map(([name, value, isRate]) => (
                                        <tr key={name}>
                                            <td>{name}</td>
                                            <td className={`metric-value ${isRate ? getMetricColor(value) : ''}`}>
                                                {isRate ? `${(value * 100).toFixed(1)}%` : value}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {/* By Source */}
                    {results.by_source && Object.keys(results.by_source).length > 0 && (
                        <div className="card" style={{ marginBottom: '20px' }}>
                            <h3 style={{ marginBottom: '12px', fontSize: '1rem' }}>By Source Dataset</h3>
                            <SourceMetricsTable data={results.by_source} labelKey="Source" tagColor="tag-blue" />
                        </div>
                    )}

                    {/* Per-question details */}
                    {results.questions && results.questions.length > 0 && (
                        <div className="card">
                            <h3 style={{ marginBottom: '12px', fontSize: '1rem' }}>
                                Per-Question Details ({results.questions.length} questions)
                            </h3>
                            <QuestionsTable
                                questions={results.questions}
                                onSelectQuestion={setSelectedQuestion}
                                maxHeight="400px"
                                showType={false}
                            />
                        </div>
                    )}
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
