import React, { useState } from 'react';
import Login from './Login';
import VoiceAgentConverted from './VoiceAgentConverted';
import DocumentQAStep from './DocumentQAStep';
import './App.css';

function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [step, setStep] = useState('login'); // 'login' | 'doc' | 'assistant'

  const handleLogin = () => {
    setLoggedIn(true);
    setStep('doc');
  };

  const renderStep = () => {
    if (!loggedIn || step === 'login') {
      return <Login onLogin={handleLogin} />;
    }
    if (step === 'doc') {
      return <DocumentQAStep onDone={() => setStep('assistant')} />;
    }
    return <VoiceAgentConverted />;
  };

  return (
    <div className="App">
      {renderStep()}
    </div>
  );
}

export default App;
