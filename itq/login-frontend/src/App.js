import React, { useState } from 'react';
import Login from './Login';
import VoiceAgentConverted from './VoiceAgentConverted';
import './App.css';

function App() {
  const [loggedIn, setLoggedIn] = useState(false);

  const handleLogin = () => {
    setLoggedIn(true);
  };

  return (
    <div className="App">
      {loggedIn ? <VoiceAgentConverted /> : <Login onLogin={handleLogin} />}
    </div>
  );
}

export default App;
