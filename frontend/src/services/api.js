import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: { 'Content-Type': 'application/json' },
});

// --- Retrieval (single query — chatbot mode) ---

export const retrieveAPI = {
    retrieve: (question, trueAnswer = null, trueContext = null) =>
        api.post('/retrieve', {
            question,
            true_answer: trueAnswer,
            true_context: trueContext,
        }),
};

// --- Generation (chatbot mode: retrieval + domain generator) ---

export const generateAPI = {
    generate: (question, generator = 'educational') =>
        api.post('/generate', { question, generator }),
};

// --- Batch Evaluation (streaming NDJSON) ---
// Uses native fetch instead of axios so we can read the response as a stream.
export const evaluateAPI = {
    /**
     * Stream batch evaluation results.
     * @param {object[]} queries  - Array of QueryItem objects
     * @param {number}   limit    - How many to run
     * @param {function} onResult - Called for each retrieved result (includes _progress)
     * @param {function} onDone   - Called once with { output_file, count }
     * @param {AbortSignal} signal
     */
    streamBatch: async (queries, limit, onResult, onDone, signal) => {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ queries, limit }),
            signal,
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
                    if (data._done) onDone(data);
                    else onResult(data);
                } catch { /* skip malformed line */ }
            }
        }
    },
};

// --- Documents ---

export const docsAPI = {
    getDocument: (docId) => api.get(`/docs/${docId}`),
};

// --- Experiments ---

export const experimentsAPI = {
    list: () => api.get('/experiments'),
    get: (name) => api.get(`/experiments/${name}`),
    create: (name, config) => api.post('/experiments', { name, config }),
    update: (name, config) => api.put(`/experiments/${name}`, { config }),
    delete: (name) => api.delete(`/experiments/${name}`),
};

// --- Health ---

export const healthAPI = {
    check: () => api.get('/health'),
};

export default api;
