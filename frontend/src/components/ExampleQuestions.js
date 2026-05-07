import React from 'react';
import { EXAMPLES, QUERY_TYPE_LABELS, QUERY_TYPE_COLORS } from '../constants/examples';

export default function ExampleQuestions({ onExampleClick }) {
  return (
    <div className="examples-panel">
      <h3 className="examples-title">Try an example</h3>
      <div className="examples-list">
        {EXAMPLES.map((ex, i) => {
          const color = QUERY_TYPE_COLORS[ex.type] || '#2563eb';
          const label = QUERY_TYPE_LABELS[ex.type] || ex.type;
          return (
            <button
              key={i}
              className="example-card"
              onClick={() => onExampleClick(ex)}
            >
              <span className="example-badge" style={{ background: color + '18', color, border: `1px solid ${color}44` }}>
                {label}
              </span>
              <p className="example-label">{ex.label}</p>
              <p className="example-preview">{ex.question.slice(0, 80)}…</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
