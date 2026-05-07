import React, { useState } from 'react';
import { QUERY_TYPE_LABELS, QUERY_TYPE_COLORS } from '../constants/examples';

const DISCLAIMER = 'Please talk to your doctor or nurse before making any decisions about your treatment.';

function cleanText(text) {
  if (!text) return '';
  return text
    .replace(/#{1,6}\s+/g, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/`{1,3}(.*?)`{1,3}/gs, '$1')
    .trim();
}

function renderPatientAnswer(text) {
  const cleaned = cleanText(text);
  const blocks = cleaned.split(/\n\n+/).filter(b => b.trim());
  return blocks.map((block, i) => {
    const isDisclaimer = block.includes('Please talk to your doctor') || block.includes('doctor or nurse');
    if (isDisclaimer) {
      return (
        <p key={i} className="patient-disclaimer">{block}</p>
      );
    }
    if (i === 0) {
      return <p key={i} className="answer-para answer-para--lead">{block}</p>;
    }
    return <p key={i} className="answer-para">{block}</p>;
  });
}

function renderDoctorAnswer(text) {
  const cleaned = cleanText(text);
  const blocks = cleaned.split(/\n\n+/).filter(b => b.trim());
  return blocks.map((block, i) => {
    const lines = block.split('\n');
    const isNumbered = lines.length > 1 && lines.every(l => /^\d+[\.\)]\s/.test(l.trim()) || l.trim() === '');
    if (isNumbered) {
      return (
        <ol key={i} className="answer-list">
          {lines.filter(l => l.trim()).map((l, j) => (
            <li key={j}>{l.replace(/^\d+[\.\)]\s+/, '')}</li>
          ))}
        </ol>
      );
    }
    return <p key={i} className="answer-para">{block}</p>;
  });
}

export default function AnswerDisplay({ patientAnswer, doctorAnswer, error, meta }) {
  const [audience, setAudience] = useState('patient');
  const [sourcesOpen, setSourcesOpen] = useState(false);

  if (error) {
    return (
      <div className="answer-card answer-card--error">
        <p className="answer-error">{error}</p>
      </div>
    );
  }

  if (!patientAnswer && !doctorAnswer) return null;

  const { query_type, language, sources = [], latency_ms, model_used } = meta || {};
  const typeColor = QUERY_TYPE_COLORS[query_type] || '#2563eb';
  const typeLabel = QUERY_TYPE_LABELS[query_type] || query_type;

  const isPatient = audience === 'patient';
  const cardStyle = isPatient
    ? { background: '#fffdf9', borderLeft: '3px solid #16a34a' }
    : { background: '#f8fafc', borderLeft: '3px solid #1e3a5f' };

  return (
    <div className="answer-card" style={cardStyle}>

      {/* Audience toggle */}
      <div className="audience-toggle">
        <button
          className={`audience-btn${isPatient ? ' audience-btn--active' : ''}`}
          onClick={() => setAudience('patient')}
        >
          ♥ Patient View
        </button>
        <button
          className={`audience-btn${!isPatient ? ' audience-btn--active audience-btn--doctor' : ''}`}
          onClick={() => setAudience('doctor')}
        >
           clipboard Clinical Reference
        </button>
      </div>

      {/* Audience label */}
      <div className="audience-label">
        {isPatient
          ? <span className="audience-label--patient">♥ Written for patients</span>
          : <span className="audience-label--doctor">📋 Clinical reference</span>
        }
      </div>

      {/* Meta pills */}
      <div className="answer-meta">
        {query_type && (
          <span className="meta-pill" style={{ background: typeColor + '18', color: typeColor, border: `1px solid ${typeColor}44` }}>
            {typeLabel}
          </span>
        )}
        {language && (
          <span className="meta-pill meta-pill--neutral">{language.toUpperCase()}</span>
        )}
        {latency_ms && (
          <span className="meta-pill meta-pill--neutral">{(latency_ms / 1000).toFixed(1)}s</span>
        )}
        {model_used && (
          <span className="meta-pill meta-pill--neutral">{model_used}</span>
        )}
      </div>

      {/* Answer body */}
      <div className="answer-body">
        {isPatient
          ? renderPatientAnswer(patientAnswer || '')
          : renderDoctorAnswer(doctorAnswer || '')
        }
      </div>

      {/* Sources */}
      {sources.length > 0 && (
        <div className="sources-section">
          <button className="sources-toggle" onClick={() => setSourcesOpen(o => !o)}>
            {sourcesOpen ? '▾' : '▸'} {sources.length} source{sources.length !== 1 ? 's' : ''}
          </button>
          {sourcesOpen && (
            <div className="sources-list">
              {sources.map((s, i) => (
                <div key={i} className="source-item">
                  <span className="source-doc">{s.document || s.doc_id || `Source ${i + 1}`}</span>
                  <span className="source-score">score: {s.score?.toFixed(3)}</span>
                  <p className="source-preview">{s.text?.slice(0, 150)}…</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
