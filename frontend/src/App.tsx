import { Switch, Route, Router as WouterRouter } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "next-themes";
import NotFound from "@/pages/not-found";
import { Layout } from "@/components/Layout/Layout";
import { ProfileProvider } from "@/context/profile-context";

// Pages
import Home from "@/pages/Dashboard/Dashboard";
import GeneratePaper from "@/pages/Examiner/Generate/Generate";
import Materials from "@/pages/TrainingStudio/Materials/Materials";
import Syllabus from "@/pages/TrainingStudio/Syllabus/Syllabus";
import History from "@/pages/Examiner/History/History";
import Settings from "@/pages/Settings/Settings";
import KnowledgeBase from "@/pages/TrainingStudio/KnowledgeBase/KnowledgeBase";
import Learning from "@/pages/Learning/Learning";
import Review from "@/pages/Examiner/Review/Review";
import LivePreview from "@/pages/Examiner/LivePreview/LivePreview";
import Analytics from "@/pages/Analytics/Analytics";
import Login from "@/pages/Login/Login";

const queryClient = new QueryClient();

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/generate" component={GeneratePaper} />
      <Route path="/materials" component={Materials} />
      <Route path="/knowledge" component={KnowledgeBase} />
      <Route path="/syllabus" component={Syllabus} />
      <Route path="/history" component={History} />
      <Route path="/settings" component={Settings} />
      
      {/* New Pages */}
      <Route path="/learning" component={Learning} />
      <Route path="/review" component={Review} />
      <Route path="/preview" component={LivePreview} />
      <Route path="/analytics" component={Analytics} />
      <Route path="/login" component={Login} />
      
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <ProfileProvider>
            <WouterRouter base={import.meta.env.BASE_URL?.replace(/\/$/, "") || ""}>
              <Layout>
                <Router />
              </Layout>
            </WouterRouter>
          </ProfileProvider>
          <Toaster richColors position="top-right" />
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;
