import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import ConversationArea from './components/ConversationArea';
import InputBar from './components/InputBar';
import { askDual, checkHealth } from './services/api';

const HISTORY_KEY = 'oncologyrag_history';

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
}

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [history, setHistory]             = useState(loadHistory);
  const [activeId, setActiveId]           = useState(null);
  const [question, setQuestion]           = useState('');
  const [selectedType, setSelectedType]   = useState('qa');
  const [selectedLang, setSelectedLang]   = useState('en');
  const [isLoading, setIsLoading]         = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [sidebarOpen, setSidebarOpen]     = useState(window.innerWidth > 768);

  useEffect(() => {
    checkHealth()
      .then(r => setBackendStatus(r.ok ? 'online' : 'offline'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  const topbarTitle = conversations.length > 0
    ? conversations[0].question.slice(0, 40) + (conversations[0].question.length > 40 ? '…' : '')
    : 'New conversation';

  async function handleSubmit() {
    if (!question.trim() || isLoading) return;
    const q = question.trim();
    setQuestion('');
    setIsLoading(true);

    const result = await askDual(q, selectedType, selectedLang);
    setIsLoading(false);

    if (result.ok) {
      const d = result.data;
      const conv = {
        id:            Date.now().toString(),
        question:      q,
        queryType:     selectedType,
        language:      selectedLang,
        patientAnswer: d.patient_answer,
        doctorAnswer:  d.doctor_answer,
        sources:       d.sources || [],
        latency_ms:    d.latency_ms,
        iterations:    d.agent_iterations,
        timestamp:     Date.now(),
      };
      setConversations(prev => [...prev, conv]);
      setActiveId(conv.id);

      const updated = [conv, ...history].slice(0, 10);
      setHistory(updated);
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(updated)); } catch {}
    } else {
      const errConv = {
        id:            Date.now().toString(),
        question:      q,
        queryType:     selectedType,
        language:      selectedLang,
        patientAnswer: `Error: ${result.error || 'Something went wrong. Is the backend running?'}`,
        doctorAnswer:  `Error: ${result.error || 'Something went wrong. Is the backend running?'}`,
        sources:       [],
        latency_ms:    null,
        iterations:    null,
        timestamp:     Date.now(),
      };
      setConversations(prev => [...prev, errConv]);
    }
  }

  function handleNewChat() {
    setConversations([]);
    setActiveId(null);
    setQuestion('');
  }

  function handleHistoryClick(item) {
    setConversations([item]);
    setActiveId(item.id);
    setSelectedType(item.queryType);
  }

  function handlePillClick(pill) {
    setSelectedType(pill.type);
    setQuestion(pill.question);
  }

  return (
    <div className="app-shell">
      <div className={sidebarOpen ? 'sidebar' : 'sidebar sidebar--collapsed'}>
        <Sidebar
          history={history}
          activeId={activeId}
          onNewChat={handleNewChat}
          onHistoryClick={handleHistoryClick}
          backendStatus={backendStatus}
        />
      </div>

      <div className="main-area">
        <TopBar
          title={topbarTitle}
          onToggleSidebar={() => setSidebarOpen(o => !o)}
        />
        <ConversationArea
          conversations={conversations}
          isLoading={isLoading}
          onPillClick={handlePillClick}
        />
        <InputBar
          question={question}
          onQuestionChange={setQuestion}
          selectedType={selectedType}
          onTypeChange={setSelectedType}
          selectedLanguage={selectedLang}
          onLanguageChange={setSelectedLang}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
