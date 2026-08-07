import { createRoot } from 'react-dom/client';

import App from './App';
import { initAPI } from '@/lib/aion-api';

import './index.css';

initAPI();

createRoot(document.getElementById('root')!).render(<App />);
