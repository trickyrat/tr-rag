import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ScrollArea } from "@/components/ui/scroll-area"
import { m } from '@/i18n';

interface FontListProps {
  fonts: { family: string }[];
  selectedFont: string | null;
  onSelectFont: (family: string) => void;
  loading?: boolean;
}

export function FontList({ fonts, selectedFont, onSelectFont, loading }: FontListProps) {
  const [search, setSearch] = useState('');

  const filteredFonts = fonts.filter(font =>
    font.family.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full border-r">
      <div className="p-4 border-b">
        <Input
          placeholder={m.fontList_searchPlaceholder()}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full"
        />
        <div className="text-xs text-muted-foreground mt-2">
          {m.fontList_count({ count: filteredFonts.length })}
        </div>
      </div>
      <ScrollArea className="flex-1 min-h-0 rounded-md border">
        <div className="p-2 space-y-1">
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">{m.fontList_loading()}</div>
          ) : (
            filteredFonts.map((font) => (
              <Button
                key={font.family}
                variant="ghost"
                className={cn(
                  "w-full justify-start font-normal text-left",
                  selectedFont === font.family && "bg-accent text-accent-foreground"
                )}
                onClick={() => onSelectFont(font.family)}
                style={{ fontFamily: font.family }}
              >
                <span className="truncate">{font.family}</span>
              </Button>
            ))
          )}
          {!loading && filteredFonts.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">{m.fontList_noResults()}</div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}