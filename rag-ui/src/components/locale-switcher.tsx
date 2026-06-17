// src/components/LocaleSwitcher.tsx
import { useLocale, setLocale } from '@/i18n';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Languages } from 'lucide-react';
import type { Locale } from "@/paraglide/runtime";

const locales: { code: Locale; name: string }[] = [
    { code: 'en', name: 'English' },
    { code: 'zh', name: '中文-简体' },
];

export function LocaleSwitcher() {
    const currentLocale = useLocale();

    return (
        <DropdownMenu>
            <DropdownMenuTrigger render={
                <Button variant="outline" size="icon">
                    <Languages className="h-4 w-4" />
                </Button>
            }>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
                {locales.map(({ code, name }) => (
                    <DropdownMenuItem
                        key={code}
                        onClick={() => setLocale(code)}
                        className={currentLocale === code ? 'bg-accent' : ''}
                    >
                        {name}
                        {currentLocale === code && (
                            <span className="ml-auto">✓</span>
                        )}
                    </DropdownMenuItem>
                ))}
            </DropdownMenuContent>
        </DropdownMenu>
    );
}