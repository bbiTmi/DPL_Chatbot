import React from 'react';
import { useNavigate } from 'react-router-dom';
import './MainPage.css';

const MainPage = () => {
  const navigate = useNavigate();

  return (
    <div className="main-container">
      <h1 className="main-title">Welcome to the Coffee Shop System</h1>
      <p className="main-subtitle">Who are you?</p>

      <div className="button-group">
        <button onClick={() => navigate('/customer')}>
          I'm a Customer
        </button>
        <button onClick={() => navigate('/manager')}>
          I'm a Manager
        </button>
      </div>
    </div>
  );
};

export default MainPage;