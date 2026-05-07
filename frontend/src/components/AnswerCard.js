import React, { useState } from 'react';
import SourcesPanel from './SourcesPanel';

function cleanText(text) {
  if (!text) return '';
  return text
    .replace(/#{1,6}\s+/g, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/`{1,3}(.*?)`{1,3}/gs, '$1')
    .trim();
}

function renderPatient(text) {
  const blocks = cleanText(text).split(/\n\n+/).filter(b => b.trim());
  return blocks.map((block, i) => {
    const isDisclaimer = block.toLowerCase().includes('talk to your doctor') ||
                         block.toLowerCase().includes('doctor or nurse');
    if (isDisclaimer) return <p key={i} className="patient-disclaimer">{block}</p>;
    if (i === 0)      return <p key={i} className="answer-para answer-para--lead">{block}</p>;
    return <p key={i} className="answer-para">{block}</p>;
  });
}

function renderDoctor(text) {
  const blocks = cleanText(text).split(/\n\n+/).filter(b => b.trim());
  return blocks.map((block, i) => {
    const lines = block.split('\n').filter(l => l.trim());
    const isNumbered = lines.length > 1 && lines.every(l => /^\d+[\.\)]\s/.test(l.trim()));
    if (isNumbered) {
      return (
        <ol key={i} className="answer-list">
          {lines.map((l, j) => <li key={j}>{l.replace(/^\d+[\.\)]\s+/, '')}</li>)}
        </ol>
      );
    }
    return <p key={i} className="answer-para">{block}</p>;
  });
}

export default function AnswerCard({ patientAnswer, doctorAnswer, sources = [], latency_ms, iterations }) {
  const [audience, setAudience] = useState('patient');
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [copied, setCopied]     = useState(false);

  const text = audience === 'patient' ? patientAnswer : doctorAnswer;

  function handleCopy() {
    navigator.clipboard.writeText(cleanText(text || '')).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="ai-card-wrap">
      <div className="ai-card">
        {/* Header: identity + audience toggle */}
        <div className="ai-card-header">
          <div className="ai-card-identity">
            <div className="ai-avatar">O</div>
            <span className="ai-label">Med Reasoning</span>
          </div>
          <div className="audience-toggle">
            <button
              className={`aud-btn${audience === 'patient' ? ' aud-btn--active' : ''}`}
              onClick={() => setAudience('patient')}
            >
              ♥ Patient
            </button>
            <button
              className={`aud-btn${audience === 'doctor' ? ' aud-btn--active' : ''}`}
              onClick={() => setAudience('doctor')}
            >
              📋 Doctor
            </button>
          </div>
        </div>

        {/* Answer text */}
        <div className="answer-text-area">
          {audience === 'patient' ? renderPatient(patientAnswer || '') : renderDoctor(doctorAnswer || '')}
        </div>

        {/* Sources panel inline */}
        {sourcesOpen && <SourcesPanel sources={sources} />}

        {/* Action row */}
        <div className="ai-card-actions">
          <div className="action-left">
            <button
              className={`action-btn${sourcesOpen ? ' action-btn--active' : ''}`}
              onClick={() => setSourcesOpen(o => !o)}
            >
              📄 Sources {sources.length > 0 && <span className="meta-pill">{sources.length}</span>}
            </button>
            {latency_ms && (
              <span className="meta-pill">{(latency_ms / 1000).toFixed(1)}s</span>
            )}
            {iterations > 1 && (
              <span className="meta-pill">{iterations} iter</span>
            )}
          </div>
          <div className="action-right">
            <button className="action-btn" onClick={handleCopy}>
              {copied ? '✓ Copied' : '⎘ Copy'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
