import { FileText, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "next-themes";
import { AIONStatusBar } from "@/components/AIONStatusBar";

export function Header() {
  const { theme, setTheme } = useTheme();
  return (
    <header className="h-16 border-b bg-card flex items-center justify-between px-6 shrink-0 gap-4">
      <div className="flex items-center gap-2 md:hidden">
        <div className="h-8 w-8 bg-primary rounded-md flex items-center justify-center">
          <FileText className="h-5 w-5 text-primary-foreground" />
        </div>
        <h1 className="font-bold text-lg text-foreground">AION Portal</h1>
      </div>
      <div className="flex-1 flex items-center justify-start md:justify-center">
        <AIONStatusBar />
      </div>
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
  );
}
