/**
 * Reusable per-question table with clickable rows.
 *
 * Props:
 *   questions: array - list of question detail objects
 *   onSelectQuestion: function - callback when a row is clicked
 *   maxHeight: string - max height for scrollable container (default: '500px')
 *   showType: boolean - whether to show question_type column (default: true)
 */
export default function QuestionsTable({ questions, onSelectQuestion, maxHeight = '500px', showType = true }) {
    if (!questions || questions.length === 0) return null;

    return (
        <div style={{ maxHeight, overflowY: 'auto' }}>
            <table className="metrics-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Question</th>
                        <th>Hit</th>
                        <th>MRR</th>
                        <th>Judge</th>
                        <th>Source</th>
                        {showType && <th>Type</th>}
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {questions.map((q, i) => (
                        <tr
                            key={q.question_id || i}
                            className="clickable-row"
                            onClick={() => onSelectQuestion(q)}
                        >
                            <td>{i + 1}</td>
                            <td style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {q.question}
                            </td>
                            <td>{q.hit ? '✅' : '❌'}</td>
                            <td className="metric-value">{q.mrr?.toFixed(2)}</td>
                            <td>{q.judge_pass === null ? '⚠️' : q.judge_pass ? '✅' : '❌'}</td>
                            <td><span className="tag tag-blue">{q.source_dataset}</span></td>
                            {showType && <td><span className="tag tag-green">{q.question_type}</span></td>}
                            <td>{q.total_time?.toFixed(1)}s</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
