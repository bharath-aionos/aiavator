import React, { useState } from 'react';

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
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      {/* Global app title shown on landing */}
      <div style={{ position: 'absolute', top: 24, left: 24, color: 'white', fontWeight: 700, fontSize: 18, opacity: 0.95 }}>
        ITQ TravelPort Smartpoint Tutor
      </div>
  <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', width: '300px', background: 'rgba(255, 255, 255, 0.15)', backdropFilter: 'blur(20px)', borderRadius: '30px', padding: '50px 40px', boxShadow: '0 25px 50px rgba(0, 0, 0, 0.25)', border: '1px solid rgba(255, 255, 255, 0.2)' }}>
        <h2 style={{ color: 'white', fontSize: '32px', fontWeight: '700', marginBottom: '10px', textAlign: 'center', textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)' }}>Login</h2>
        <p style={{ color: 'white', textAlign: 'center', marginBottom: '20px' }}>Default Username: admin<br />Default Password: password</p>
        <label style={{ color: 'white', marginBottom: '10px' }}>
          Username:
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={{ marginBottom: '10px', padding: '8px', width: '100%', borderRadius: '10px', border: 'none', background: 'rgba(255, 255, 255, 0.2)', color: 'white' }}
          />
        </label>
        <label style={{ color: 'white', marginBottom: '10px' }}>
          Password:
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ marginBottom: '10px', padding: '8px', width: '100%', borderRadius: '10px', border: 'none', background: 'rgba(255, 255, 255, 0.2)', color: 'white' }}
          />
        </label>
        <button type="submit" style={{ padding: '10px', background: 'linear-gradient(135deg, #ff6b6b, #ee5a52)', color: 'white', border: 'none', cursor: 'pointer', borderRadius: '10px', fontWeight: '600' }}>
          Login
        </button>
      </form>
      {/* Bottom-right AIONOS logo (place a transparent PNG at /public/aionos-logo.png) */}
      <div style={{ position: 'fixed', right: 18, bottom: 18, zIndex: 1000 }}>
        <img src="/aionos-logo.png" alt="By AIONOS" style={{ width: 140, height: 'auto', display: 'block', filter: 'drop-shadow(0 2px 6px rgba(0,0,0,0.3))' }} />
      </div>
    </div>
  );
}

export default Login;
