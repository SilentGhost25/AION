import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useLocation } from "wouter";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [location] = useLocation();
  const isLoginPage = location === "/login";

  if (isLoginPage) {
    return (
      <div className="flex h-screen bg-background overflow-hidden">
        <main className="flex-1 overflow-y-auto bg-muted/20">
          <div className="p-6 md:p-8 w-full max-w-full mx-auto h-full flex items-center justify-center">
            {children}
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto bg-muted/20">
          <div className="p-6 md:p-8 max-w-7xl mx-auto h-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
