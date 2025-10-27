import React, { useState } from 'react';
import Login from './Login';
import VoiceAgentConverted from './VoiceAgentConverted';
import DocumentQAStep from './DocumentQAStep';
import './App.css';

function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [step, setStep] = useState('login'); // 'login' | 'doc' | 'assistant'

  // Shared RAG state
  const [ragQuery, setRagQuery] = useState('');
  const [ragAnswer, setRagAnswer] = useState('');
  const [ragPagesUsed, setRagPagesUsed] = useState([]);
  const [ragCandidates, setRagCandidates] = useState([]);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragNotFound, setRagNotFound] = useState(false);

  // Unified RAG query handler
  const handleAsk = async (query) => {
    if (!query.trim()) return;
    setRagLoading(true);
    setRagQuery(query);
    try {
      const res = await fetch('http://localhost:8000/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error('Query failed');
      const data = await res.json();
      const answer = data.answer || '';
      const pages = Array.isArray(data.pages) ? data.pages : [];
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      // If RAG explicitly reports not available and no pages, treat as not found and hide UI
      const lower = (answer || '').toLowerCase();
      const notAvailable = lower.includes('not available in the document') || lower.includes('not available');
      if (notAvailable && pages.length === 0) {
        setRagAnswer('');
        setRagPagesUsed([]);
        setRagCandidates([]);
        setRagNotFound(true);
      } else {
        setRagAnswer(answer);
        setRagPagesUsed(pages);
        setRagCandidates(candidates);
        setRagNotFound(false);
      }
    } catch (e) {
      setRagAnswer('');
      setRagPagesUsed([]);
      setRagCandidates([]);
      setRagNotFound(false);
    } finally {
      setRagLoading(false);
    }
  };

  // Allow backend-driven updates (e.g., voice flow writes last result to server)
  const updateRagFromServer = (data) => {
    if (!data) return;
    const answer = data.answer || '';
    const pages = Array.isArray(data.pages) ? data.pages : [];
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const lower = (answer || '').toLowerCase();
    const notAvailable = lower.includes('not available in the document') || lower.includes('not available');
    if (notAvailable && pages.length === 0) {
      setRagAnswer('');
      setRagPagesUsed([]);
      setRagCandidates([]);
      setRagNotFound(true);
    } else {
      setRagAnswer(answer);
      setRagPagesUsed(pages);
      setRagCandidates(candidates);
      setRagNotFound(false);
    }
  };

  const handleLogin = () => {
    setLoggedIn(true);
    setStep('doc');
  };

  const renderStep = () => {
    if (!loggedIn || step === 'login') {
      return <Login onLogin={handleLogin} />;
    }
    if (step === 'doc') {
      return (
        <DocumentQAStep
          onDone={() => setStep('assistant')}
          ragQuery={ragQuery}
          ragAnswer={ragAnswer}
          ragPagesUsed={ragPagesUsed}
          ragCandidates={ragCandidates}
          ragLoading={ragLoading}
          handleAsk={handleAsk}
          updateRagFromServer={updateRagFromServer}
        />
      );
    }
    return (
      <VoiceAgentConverted
        ragQuery={ragQuery}
        ragAnswer={ragAnswer}
        ragPagesUsed={ragPagesUsed}
        ragCandidates={ragCandidates}
        ragLoading={ragLoading}
        handleAsk={handleAsk}
        updateRagFromServer={updateRagFromServer}
      />
    );
  };

  return (
    <div className="App">
      {renderStep()}
    </div>
  );
}

export default App;
