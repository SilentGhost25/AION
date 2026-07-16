import { FileText, History, Home, Settings, BookOpen, BookMarked, Sparkles, Database, Moon, Sun } from "lucide-react";
import { Link, useLocation } from "wouter";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useTheme } from "next-themes";
import { useProfile } from "@/context/profile-context";

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { href: "/", icon: Home, label: "Home" },
  { href: "/generate", icon: Sparkles, label: "Generate Paper" },
  { href: "/materials", icon: BookOpen, label: "Study Materials" },
  { href: "/knowledge", icon: Database, label: "Knowledge Base" },
  { href: "/syllabus", icon: BookMarked, label: "Syllabus" },
  { href: "/history", icon: History, label: "Paper Review" },
  { href: "/settings", icon: Settings, label: "Settings" },
];

export function Layout({ children }: LayoutProps) {
  const [location] = useLocation();
  const { theme, setTheme } = useTheme();
  const { profile } = useProfile();

  const initials = profile.name.split(" ").map(n => n[0]).slice(0, 2).join("");

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar */}
      <aside className="hidden md:flex w-64 flex-col border-r bg-card shadow-sm">
        <div className="p-6 flex items-center gap-3">
          <div className="h-8 w-8 bg-primary rounded-md flex items-center justify-center">
            <FileText className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight tracking-tight text-foreground">QP Generator</h1>
            <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">DSATM Faculty Portal</p>
          </div>
        </div>
        <Separator />
        <ScrollArea className="flex-1 py-4">
          <nav className="space-y-1 px-4">
            {navItems.map((item) => (
              <Link key={item.href} href={item.href}>
                <span
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors cursor-pointer",
                    location === item.href
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </span>
              </Link>
            ))}
          </nav>
        </ScrollArea>
        <div className="p-4 border-t">
          <div className="flex items-center gap-3 px-2 py-2">
            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-sm font-semibold text-primary">{initials}</span>
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-medium truncate">{profile.name}</span>
              <span className="text-xs text-muted-foreground truncate">{profile.designation}</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-16 border-b bg-card flex items-center justify-between px-6 shrink-0">
          <div className="flex items-center gap-2 md:hidden">
            <div className="h-8 w-8 bg-primary rounded-md flex items-center justify-center">
              <FileText className="h-5 w-5 text-primary-foreground" />
            </div>
            <h1 className="font-bold text-lg text-foreground">QP Generator</h1>
          </div>
          <div className="hidden md:flex flex-1" />
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto bg-muted/20">
          <div className="p-6 md:p-8 max-w-7xl mx-auto h-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
