import { Route, Switch, Router as WouterRouter } from 'wouter';
import Wizard from '@/pages/Wizard';
import NotFound from '@/pages/not-found';
import { Toaster } from '@/components/ui/toaster';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';

const queryClient = new QueryClient();

function Router() {
  return (
    <Switch>
      <Route path="/" component={Wizard} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
        <Router />
      </WouterRouter>
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
