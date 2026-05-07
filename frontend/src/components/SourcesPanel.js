import React from 'react';

export default function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-panel">
      {sources.map((s, i) => (
        <div key={i} className="source-item">
          <div className="source-doc-name">
            {s.document || s.doc_id || `Source ${i + 1}`}
          </div>
          <p className="source-preview-text">
            {s.text?.slice(0, 120)}{s.text?.length > 120 ? '…' : ''}
          </p>
          <div className="source-score-bar">
            <div
              className="source-score-fill"
              style={{ width: `${Math.min((s.score || 0) * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
