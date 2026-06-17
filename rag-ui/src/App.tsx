import { useState } from 'react';
import { useSystemFonts } from '@/hooks/useSystemFonts';
import { FontList } from '@/components/font-list';
import { FontPreview } from '@/components/font-preview';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ThemeProvider } from '@/components/theme-provider';
import { ModeToggle } from '@/components/mode-toggle';
import { LocaleSwitcher } from '@/components/locale-switcher';
import { m, useLocale } from "@/i18n";

function App() {
  const { fonts, loading, error, isApiSupported, isAuthorized, loadFonts, useFallbackFonts } = useSystemFonts();
  const [selectedFont, setSelectedFont] = useState<string | null>(null);
  const [hasAttemptedLoad, setHasAttemptedLoad] = useState(false);

  const handleLoadFonts = () => {
    loadFonts();
    setHasAttemptedLoad(true);
  };

  const handleUseFallback = () => {
    useFallbackFonts();
    setHasAttemptedLoad(true);
  };

  if (!hasAttemptedLoad && isApiSupported) {
    return (
      <div className="h-screen flex items-center justify-center p-4">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle>{m.fontPreviewer_title()}</CardTitle>
            <CardDescription>
              {m.fontPreviewer_description()}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={handleLoadFonts} className="w-full">
              {m.fontPreviewer_authorizeButton()}
            </Button>
            <Button variant="outline" onClick={handleUseFallback} className="w-full">
              {m.fontPreviewer_fallbackButton()}
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Laoding status
  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-muted-foreground">{m.fontPreviewer_loading()}</div>
      </div>
    );
  }

  // Main
  return (
    <ThemeProvider>
      <div className="h-screen flex flex-col bg-muted/30">
        <header className="border-b bg-background/80 backdrop-blur-sm sticky top-0 z-10">
          <div className="flex justify-between items-center px-4 py-2">
            <div className="flex flex-col gap-1">
              <h1 className="text-2xl font-bold">{m.fontPreviewer_appTitle()}</h1>
              <p className="text-muted-foreground text-sm">
                {isApiSupported
                  ? isAuthorized
                    ? m.fontPreviewer_authorizedStatus()
                    : m.fontPreviewer_fallbackStatus()
                  : m.fontPreviewer_unsupportedStatus()}
              </p>
              {!isAuthorized && isApiSupported && (
                <Button variant="outline" onClick={handleLoadFonts} className="mt-1">
                  {m.fontPreviewer_reauthorizeButton()}
                </Button>
              )}
            </div>

            <div className="flex items-center gap-2">
              <ModeToggle />
              <LocaleSwitcher />
            </div>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          <div className="w-80 flex flex-col h-full">
            <FontList
              fonts={fonts}
              selectedFont={selectedFont}
              onSelectFont={setSelectedFont}
              loading={loading}
            />
          </div>
          <div className="flex-1 p-6 overflow-auto">
            {selectedFont ? (
              <FontPreview fontFamily={selectedFont} />
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                {m.fontPreviewer_selectPrompt()}
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="absolute bottom-4 left-4 right-4 z-10">
            <Alert variant="destructive">
              <AlertTitle>{m.fontPreviewer_alertTitle()}</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        )}
      </div>
    </ThemeProvider>

  );
}

export default App;