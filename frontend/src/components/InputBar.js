import React, { useRef } from 'react';
import { QUERY_TYPE_LABELS, PLACEHOLDERS } from '../constants/examples';

const QT_ICONS = {
  qa: '?', mcqa: '☑', lfqa: '≡', fact_check: '✓', fill_blank: '_', jeopardy: '!',
};
const TYPES = ['qa', 'mcqa', 'lfqa', 'fact_check', 'fill_blank', 'jeopardy'];
const LANGUAGES = [
  { value: 'en',      label: 'English'  },
  { value: 'hi',      label: 'Hindi'    },
  { value: 'ta',      label: 'Tamil'    },
  { value: 'te',      label: 'Telugu'   },
  { value: 'hinglish', label: 'Hinglish' },
];

export default function InputBar({
  question, onQuestionChange,
  selectedType, onTypeChange,
  selectedLanguage, onLanguageChange,
  onSubmit, isLoading,
}) {
  const textareaRef = useRef(null);
  const placeholder = PLACEHOLDERS[selectedType] || PLACEHOLDERS.qa;

  function handleKeyDown(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isLoading && question.trim()) onSubmit();
    }
  }

  function handleChange(e) {
    onQuestionChange(e.target.value);
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    }
  }

  return (
    <div className="input-bar">
      <div className="input-bar-inner">

        {/* Query type tabs */}
        <div className="qt-tabs">
          {TYPES.map(type => (
            <button
              key={type}
              className={`qt-tab${selectedType === type ? ' qt-tab--active' : ''}`}
              onClick={() => onTypeChange(type)}
            >
              <span>{QT_ICONS[type]}</span>
              {QUERY_TYPE_LABELS[type]}
            </button>
          ))}
        </div>

        {/* Textarea — clean, only send button inside */}
        <div className="textarea-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder={placeholder}
            value={question}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
          <button
            className="send-btn"
            onClick={onSubmit}
            disabled={isLoading || !question.trim()}
            aria-label="Send"
          >
            ➤
          </button>
        </div>

        {/* Controls row below textarea */}
        <div className="input-controls-row">
          <div className="lang-selector-wrap">
            <span className="lang-globe">🌐</span>
            <select
              className="lang-select"
              value={selectedLanguage}
              onChange={e => onLanguageChange(e.target.value)}
              disabled={isLoading}
            >
              {LANGUAGES.map(l => (
                <option key={l.value} value={l.value}>{l.label}</option>
              ))}
            </select>
          </div>
          <div className="input-hints">
            <span className="input-hint-text">Ctrl+Enter to send</span>
            <span className="input-hint-text">{question.length} chars</span>
          </div>
        </div>

      </div>
    </div>
  );
}
