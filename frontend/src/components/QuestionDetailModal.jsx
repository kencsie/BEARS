import { useState, useEffect } from 'react';
import { docsAPI } from '../services/api';

/**
 * Modal component to display detailed per-question evaluation results.
 * Supports expanding doc_ids to view document content from ChromaDB.
 *
 * Props:
 *   question: object - the question detail from eval results
 *   onClose: function - callback to close modal
 */
export default function QuestionDetailModal({ question, onClose }) {
    const [docContents, setDocContents] = useState({});
    const [docLoading, setDocLoading] = useState({});

    // Prevent body scroll when modal is open
    useEffect(() => {
        document.body.classList.add('modal-open');
        return () => {
            document.body.classList.remove('modal-open');
        };
    }, []);

    if (!question) return null;

    async function loadDocument(docId) {
        if (docContents[docId]) {
            // Toggle close
            const updated = { ...docContents };
            delete updated[docId];
            setDocContents(updated);
            return;
        }

        setDocLoading(prev => ({ ...prev, [docId]: true }));
        try {
            const res = await docsAPI.getDocument(docId);
            setDocContents(prev => ({ ...prev, [docId]: res.data }));
        } catch (err) {
            setDocContents(prev => ({
                ...prev,
                [docId]: { error: err.response?.data?.detail || err.message },
            }));
        } finally {
            setDocLoading(prev => ({ ...prev, [docId]: false }));
        }
    }

    function renderDocIds(label, docIds, isGold = false) {
        if (!docIds || docIds.length === 0) return null;
        return (
            <div className="detail-field">
                <div className="detail-label">{label}</div>
                <div className="doc-id-list">
                    {docIds.map((docId) => {
                        const isRetrieved = !isGold && question.gold_doc_ids?.includes(docId);
                        return (
                            <div key={docId} className="doc-id-item">
                                <button
                                    className={`doc-id-btn ${isRetrieved ? 'doc-id-match' : ''}`}
                                    onClick={() => loadDocument(docId)}
                                    disabled={docLoading[docId]}
                                >
                                    {docLoading[docId] ? (
                                        <span className="loading-spinner" style={{ width: 14, height: 14 }}></span>
                                    ) : (
                                        <span className="doc-id-arrow">{docContents[docId] ? '▼' : '▶'}</span>
                                    )}
                                    <code>{docId}</code>
                                    {isRetrieved && <span className="tag tag-green" style={{ marginLeft: 6 }}>match</span>}
                                </button>
                                {docContents[docId] && (
                                    <div className="doc-content-box">
                                        {docContents[docId].error ? (
                                            <div style={{ color: 'var(--accent-red)' }}>
                                                Error: {docContents[docId].error}
                                            </div>
                                        ) : (
                                            <>
                                                {docContents[docId].metadata && (
                                                    <div className="doc-metadata">
                                                        {Object.entries(docContents[docId].metadata)
                                                            .sort(([k1], [k2]) => k1.localeCompare(k2))
                                                            .map(([k, v]) => (
                                                                <span key={k} className="doc-meta-item">
                                                                    <span style={{ color: 'var(--text-muted)' }}>{k}:</span>{' '}
                                                                    <span style={{ color: 'var(--accent-cyan)' }}>{String(v)}</span>
                                                                </span>
                                                            ))}
                                                    </div>
                                                )}
                                                <div className="doc-text">{docContents[docId].content}</div>
                                            </>
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    }

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">Question Details</h2>
                    <button className="modal-close" onClick={onClose}>✕</button>
                </div>

                <div className="modal-body">
                    {/* Status badges */}
                    <div className="detail-badges">
                        <span className={`tag ${question.hit ? 'tag-green' : 'tag-orange'}`}>
                            Hit: {question.hit ? '✅ Yes' : '❌ No'}
                        </span>
                        <span className={`tag ${question.judge_pass ? 'tag-green' : 'tag-orange'}`}>
                            Judge: {question.judge_pass ? '✅ Pass' : '❌ Fail'}
                        </span>
                        <span className="tag tag-blue">{question.source_dataset}</span>
                        <span className="tag tag-purple">{question.question_type}</span>
                    </div>

                    {/* Question & Answers */}
                    <div className="detail-field">
                        <div className="detail-label">Question</div>
                        <div className="detail-value">{question.question}</div>
                    </div>
                    <div className="detail-field">
                        <div className="detail-label">Gold Answer</div>
                        <div className="detail-value detail-gold">{question.gold_answer}</div>
                    </div>
                    <div className="detail-field">
                        <div className="detail-label">Model Answer</div>
                        <div className="detail-value detail-model">{question.model_answer}</div>
                    </div>

                    {/* Metrics */}
                    <div className="detail-metrics-row">
                        <div className="detail-metric">
                            <span className="detail-label">MRR</span>
                            <span className="metric-value">{question.mrr?.toFixed(3)}</span>
                        </div>
                        <div className="detail-metric">
                            <span className="detail-label">AP</span>
                            <span className="metric-value">{question.ap?.toFixed(3)}</span>
                        </div>
                        <div className="detail-metric">
                            <span className="detail-label">Found</span>
                            <span className="metric-value">{question.found_count}</span>
                        </div>
                        <div className="detail-metric">
                            <span className="detail-label">Retrieval</span>
                            <span className="metric-value">{question.retrieval_time?.toFixed(2)}s</span>
                        </div>
                        <div className="detail-metric">
                            <span className="detail-label">Generation</span>
                            <span className="metric-value">{question.generation_time?.toFixed(2)}s</span>
                        </div>
                        <div className="detail-metric">
                            <span className="detail-label">Total</span>
                            <span className="metric-value">{question.total_time?.toFixed(2)}s</span>
                        </div>
                    </div>

                    {/* Doc IDs */}
                    {renderDocIds('Gold Doc IDs', question.gold_doc_ids, true)}
                    {renderDocIds('Retrieved Doc IDs', question.retrieved_doc_ids, false)}
                </div>
            </div>
        </div>
    );
}
