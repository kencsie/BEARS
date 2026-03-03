/**
 * Get CSS class for metric value coloring based on thresholds.
 * @param {number} value - Metric value (0-1 range)
 * @param {number[]} thresholds - [okThreshold, goodThreshold]
 * @returns {string} CSS class name
 */
export function getMetricColor(value, thresholds = [0.5, 0.7]) {
    if (value >= thresholds[1]) return 'metric-good';
    if (value >= thresholds[0]) return 'metric-ok';
    return 'metric-bad';
}
