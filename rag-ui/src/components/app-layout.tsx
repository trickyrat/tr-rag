import { NavLink, Outlet, useLocation } from "react-router-dom";
import { ModeToggle } from "@/components/mode-toggle";
import { LocaleSwitcher } from "@/components/locale-switcher";
import { cn } from "@/lib/utils";
import { Type, BarChart3 } from "lucide-react";

export function AppLayout() {
  const loc = useLocation();

  const tabs = [
    { to: "/", label: "Font Previewer", icon: Type },
    { to: "/evaluation", label: "Evaluation", icon: BarChart3 },
  ];

  return (
    <div className="h-screen flex flex-col bg-muted/30">
      {/* Header */}
      <header className="border-b bg-background/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="flex justify-between items-center px-4 py-2">
          <div className="flex items-center gap-6">
            <h1 className="text-lg font-bold">RAG Tools</h1>
            <nav className="flex gap-1">
              {tabs.map((t) => {
                const active = t.to === "/"
                  ? loc.pathname === "/"
                  : loc.pathname.startsWith(t.to);
                return (
                  <NavLink
                    key={t.to}
                    to={t.to}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                      active
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    )}
                  >
                    <t.icon className="size-4" />
                    {t.label}
                  </NavLink>
                );
              })}
            </nav>
          </div>

          <div className="flex items-center gap-2">
            <ModeToggle />
            <LocaleSwitcher />
          </div>
        </div>
      </header>

      {/* Page content */}
      <Outlet />
    </div>
  );
}
