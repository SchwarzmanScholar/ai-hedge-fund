import React from 'react';
import ReactDOM from 'react-dom/client';

import App from './App';
import { AuthProvider } from './contexts/auth-context';
import { NodeProvider } from './contexts/node-context';
import { ThemeProvider } from './providers/theme-provider';

import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider>
      <ThemeProvider>
        <NodeProvider>
          <App />
        </NodeProvider>
      </ThemeProvider>
    </AuthProvider>
  </React.StrictMode>
);
