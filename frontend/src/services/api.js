import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: { 'Content-Type': 'application/json' },
});

// --- Evaluation ---

export const evalAPI = {
    /** Start an evaluation task */
    start: (params) => api.post('/eval/start', params),

    /** Get evaluation task status */
    getStatus: (taskId) => api.get(`/eval/status/${taskId}`),

    /** Get evaluation results */
    getResults: (taskId) => api.get(`/eval/results/${taskId}`),

    /** List past evaluation result files */
    getHistory: () => api.get('/eval/history'),

    /** Load a specific history file */
    getHistoryFile: (filename) => api.get(`/eval/history/${filename}`),

    /** List agents available for evaluation */
    getAgents: () => api.get('/eval/agents'),

    /** Get query statistics */
    getQueryStats: () => api.get('/eval/queries/stats'),
};

// --- Query ---

export const queryAPI = {
    /** Single question query */
    query: (question, agent = null) =>
        api.post('/query', { question, agent }),

    /** Health check */
    health: () => api.get('/health'),
};

// --- Experiments ---

export const experimentsAPI = {
    /** List all experiment configs */
    list: () => api.get('/experiments'),

    /** Get a specific experiment config */
    get: (name) => api.get(`/experiments/${name}`),

    /** Create a new experiment config */
    create: (name, config) => api.post('/experiments', { name, config }),

    /** Update an experiment config */
    update: (name, config) => api.put(`/experiments/${name}`, { config }),

    /** Delete an experiment config */
    delete: (name) => api.delete(`/experiments/${name}`),
};

// --- Documents ---

export const docsAPI = {
    /** Fetch document content by ID from ChromaDB */
    getDocument: (docId) => api.get(`/docs/${docId}`),
};

export default api;
