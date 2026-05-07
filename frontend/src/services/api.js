const BASE_URL = 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${BASE_URL}/api/health`);
    if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
    const data = await res.json();
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

export async function askQuestion(query, queryTypeHint = 'qa', language = 'en') {
  try {
    const res = await fetch(`${BASE_URL}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        query_type: queryTypeHint === 'fact_check' ? 'fact_verification' : queryTypeHint,
        language,
        top_k: 5,
        agentic_pattern: 'auto',
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return { ok: false, error: err.detail || `Server error ${res.status}` };
    }
    const data = await res.json();
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

export async function askDual(query, queryTypeHint = 'qa', language = 'en') {
  try {
    const res = await fetch(`${BASE_URL}/api/query/dual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        query_type: queryTypeHint === 'fact_check' ? 'fact_verification' : queryTypeHint,
        language,
        top_k: 5,
        agentic_pattern: 'auto',
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return { ok: false, error: err.detail || `Server error ${res.status}` };
    }
    const data = await res.json();
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

export async function getMetrics() {
  try {
    const res = await fetch(`${BASE_URL}/api/metrics`);
    if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
    const data = await res.json();
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}
