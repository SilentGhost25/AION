import { createRoot } from 'react-dom/client';

import App from './App';
import { initAPI } from '@/lib/aion-api';

import './index.css';
import 'katex/dist/katex.min.css';

initAPI();

createRoot(document.getElementById('root')!).render(<App />);
