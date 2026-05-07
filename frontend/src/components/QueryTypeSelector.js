import React from 'react';
import { QUERY_TYPE_LABELS } from '../constants/examples';

const TYPES = ['qa', 'mcqa', 'lfqa', 'fact_check', 'fill_blank', 'jeopardy'];

export default function QueryTypeSelector({ selectedType, onTypeChange }) {
  return (
    <div className="query-type-selector">
      {TYPES.map(type => (
        <button
          key={type}
          className={`qt-btn ${selectedType === type ? 'qt-btn--active' : ''}`}
          onClick={() => onTypeChange(type)}
        >
          {QUERY_TYPE_LABELS[type]}
        </button>
      ))}
    </div>
  );
}
