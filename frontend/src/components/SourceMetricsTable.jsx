import { getMetricColor } from '../utils/metrics';

/**
 * Reusable table for displaying source/type-based metrics.
 *
 * Props:
 *   data: object - { key: { total_questions, hit_rate, mrr, map, generation_pass_rate, avg_total_time } }
 *   labelKey: string - column header for the label (e.g. "Source", "Type")
 *   tagColor: string - CSS class for the tag (e.g. "tag-blue", "tag-green")
 *   showAvgTime: boolean - whether to show avg time column
 */
export default function SourceMetricsTable({ data, labelKey = 'Source', tagColor = 'tag-blue', showAvgTime = false }) {
    if (!data || Object.keys(data).length === 0) return null;

    return (
        <table className="metrics-table">
            <thead>
                <tr>
                    <th>{labelKey}</th>
                    <th>Questions</th>
                    <th>Hit Rate</th>
                    <th>MRR</th>
                    <th>MAP</th>
                    <th>Pass Rate</th>
                    {showAvgTime && <th>Avg Time</th>}
                </tr>
            </thead>
            <tbody>
                {Object.entries(data).map(([label, m]) => (
                    <tr key={label}>
                        <td><span className={`tag ${tagColor}`}>{label}</span></td>
                        <td>{m.total_questions}</td>
                        <td className={`metric-value ${getMetricColor(m.hit_rate)}`}>{(m.hit_rate * 100).toFixed(1)}%</td>
                        <td className={`metric-value ${getMetricColor(m.mrr)}`}>{(m.mrr * 100).toFixed(1)}%</td>
                        <td className={`metric-value ${getMetricColor(m.map)}`}>{(m.map * 100).toFixed(1)}%</td>
                        <td className={`metric-value ${getMetricColor(m.generation_pass_rate)}`}>{(m.generation_pass_rate * 100).toFixed(1)}%</td>
                        {showAvgTime && <td>{m.avg_total_time?.toFixed(2)}s</td>}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}
