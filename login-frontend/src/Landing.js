import React from 'react';
import logo from './itq-logo.png';

function Landing({ onGetStarted }) {
  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column',
      justifyContent: 'center', 
      alignItems: 'center', 
      minHeight: '100vh',
      background: '#f8fafc'
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        maxWidth: '800px',
        padding: '40px',
        textAlign: 'center'
      }}>
        <img src={logo} alt="ITQ Logo" style={{ 
          width: '80px',
          marginBottom: '32px'
        }} />
        
        <h1 style={{
          fontSize: '48px',
          fontWeight: 'bold',
          marginBottom: '16px',
          color: '#1e293b'
        }}>
          <span style={{ color: '#3b82f6' }}>ITQ</span> Travel Assistant
        </h1>

        <p style={{
          fontSize: '18px',
          color: '#64748b',
          marginBottom: '32px',
          maxWidth: '600px'
        }}>
          Powered by RAG and voice. Your Travelport knowledge, at your voice.
        </p>

        <button
          onClick={onGetStarted}
          style={{
            background: '#3b82f6',
            color: 'white',
            padding: '12px 32px',
            borderRadius: '8px',
            border: 'none',
            fontSize: '16px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'transform 0.1s ease-in-out',
            ':hover': {
              transform: 'scale(1.02)'
            }
          }}
        >
          Get Started
        </button>
      </div>
    </div>
  );
}

export default Landing;