import React, { useRef } from 'react';
import { PLACEHOLDERS } from '../constants/examples';

const LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'Hindi' },
  { value: 'ta', label: 'Tamil' },
  { value: 'te', label: 'Telugu' },
  { value: 'hinglish', label: 'Hinglish' },
];

export default function QuestionInput({
  question, onQuestionChange,
  selectedType, selectedLanguage, onLanguageChange,
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
      el.style.height = Math.min(el.scrollHeight, 240) + 'px';
    }
  }

  return (
    <div className="question-input-card">
      <textarea
        ref={textareaRef}
        className="question-textarea"
        placeholder={placeholder}
        value={question}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        rows={4}
        disabled={isLoading}
      />
      <div className="question-input-footer">
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
        <span className="input-hint">Ctrl+Enter to submit</span>
        <button
          className="ask-btn"
          onClick={onSubmit}
          disabled={isLoading || !question.trim()}
        >
          {isLoading ? 'Thinking...' : 'Ask Question'}
        </button>
      </div>
    </div>
  );
}
