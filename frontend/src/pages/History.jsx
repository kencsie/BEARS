import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { evalAPI } from '../services/api';

export default function History() {
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        loadHistory();
    }, []);

    async function loadHistory() {
        try {
            const res = await evalAPI.getHistory();
            setFiles(res.data.results);
        } catch (err) {
            console.error('Failed to load history:', err);
        } finally {
            setLoading(false);
        }
    }

    function formatDate(timestamp) {
        return new Date(timestamp * 1000).toLocaleString('zh-TW');
    }

    function formatSize(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function getAgentFromFilename(filename) {
        const name = filename.replace('.json', '');
        if (name.startsWith('hybrid')) return 'hybrid';
        if (name.startsWith('kg')) return 'kg';
        if (name.startsWith('agentic')) return 'agentic';
        if (name.startsWith('orchestrator')) return 'orchestrator';
        return name.split('_')[0];
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
                <h1 className="page-title">History</h1>
                <p className="page-subtitle">過去的評估結果</p>
            </div>

            {files.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-state-icon">📭</div>
                    <div className="empty-state-text">尚無評估結果。前往 Dashboard 啟動評估。</div>
                </div>
            ) : (
                <div>
                    {files.map((file) => {
                        const agent = getAgentFromFilename(file.filename);
                        return (
                            <div
                                className="history-item"
                                key={file.filename}
                                onClick={() => navigate(`/results/${file.filename}`)}
                            >
                                <div>
                                    <div className="history-filename">{file.filename}</div>
                                    <div className="history-meta">
                                        <span className={`tag tag-blue`} style={{ marginRight: '8px' }}>{agent}</span>
                                        {formatDate(file.modified)} · {formatSize(file.size_bytes)}
                                    </div>
                                </div>
                                <button className="btn btn-secondary btn-sm">View →</button>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
