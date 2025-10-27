import React, { useState } from 'react';
import logo from './itq-logo.png';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Handle login logic here, e.g., send to backend
    console.log('Username:', username);
    console.log('Password:', password);
    try {
      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();
      if (data.success) {
        onLogin();
      } else {
        alert(data.error || 'Login failed');
      }
    } catch (error) {
      console.error('Login error:', error);
      alert('Login failed');
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh', 
      background: '#f8fafc'
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        maxWidth: '400px',
        width: '100%',
        padding: '40px',
        background: 'white',
        borderRadius: '16px',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
      }}>
        <img src={logo} alt="ITQ Logo" style={{ width: '60px', marginBottom: '24px' }} />
        <h2 style={{ 
          fontSize: '24px', 
          fontWeight: '600',
          color: '#1e293b',
          marginBottom: '8px'
        }}>Login</h2>
        <p style={{ 
          color: '#64748b',
          marginBottom: '24px',
          fontSize: '14px'
        }}>Default: admin / password</p>
        
        <form onSubmit={handleSubmit} style={{ width: '100%' }}>
          <div style={{ marginBottom: '16px' }}>
            <label 
              htmlFor="username" 
              style={{ 
                display: 'block',
                marginBottom: '8px',
                color: '#475569',
                fontSize: '14px'
              }}
            >
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              style={{
                width: '100%',
                padding: '8px 12px',
                borderRadius: '6px',
                border: '1px solid #e2e8f0',
                fontSize: '14px'
              }}
            />
          </div>

          <div style={{ marginBottom: '24px' }}>
            <label
              htmlFor="password"
              style={{
                display: 'block',
                marginBottom: '8px',
                color: '#475569',
                fontSize: '14px'
              }}
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{
                width: '100%',
                padding: '8px 12px',
                borderRadius: '6px',
                border: '1px solid #e2e8f0',
                fontSize: '14px'
              }}
            />
          </div>

          <button
            type="submit"
            style={{
              width: '100%',
              padding: '12px',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Login
          </button>
        </form>
      </div>
      {/* Bottom-right AIONOS logo */}
      <div style={{ position: 'fixed', right: 18, bottom: 18, zIndex: 1000 }}>
        <img src="/aionos-logo.png" alt="By AIONOS" style={{ width: 140, height: 'auto', display: 'block', filter: 'drop-shadow(0 2px 6px rgba(0,0,0,0.3))' }} />
      </div>
    </div>
  );
}

export default Login;
