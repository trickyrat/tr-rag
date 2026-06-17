import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { m } from '@/i18n';

interface FontPreviewProps {
  fontFamily: string;
}

const LIGATURE_DEMO_TEXT = "=>  == === !=";

export function FontPreview({ fontFamily }: FontPreviewProps) {
  const [text, setText] = useState('The quick brown fox jumps over the lazy dog');
  const [fontSize, setFontSize] = useState<number | readonly number[]>([24]);
  const [ligaturesEnabled, setLigaturesEnabled] = useState(true);



  const insertDemoText = () => {
    setText(LIGATURE_DEMO_TEXT);
  };


  return (
    <Card className="w-full h-full">
      <CardHeader>
        <CardTitle>{m.fontPreview_title({ fontFamily })}</CardTitle>
        <div className="flex flex-col gap-4 mt-2">
          <div className="flex gap-2 items-end">
            <div className="flex-1">
              <Label htmlFor="preview-text">{m.fontPreview_previewTextLabel()}</Label>
              <Input
                id="preview-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={m.fontPreview_previewTextPlaceholder()}
              />
            </div>
            <Button
              type="button"
              variant="outline"
              onClick={insertDemoText}
              className="shrink-0"
            >
              {m.fontPreview_demoTextButton()}
            </Button>
          </div>

          <div className="flex gap-4 items-center">
            <div className="flex-1">
              <Label htmlFor="font-size">
                {m.fontPreview_fontSizeLabel({ size: fontSize })}
              </Label>
              <Slider
                id="font-size"
                min={12}
                max={72}
                step={1}
                value={fontSize}
                onValueChange={(val) => setFontSize(val)}
                className="mt-2"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="ligature-switch" className="cursor-pointer">
                {m.fontPreview_ligatureLabel()}
              </Label>
              <Switch
                id="ligature-switch"
                checked={ligaturesEnabled}
                onCheckedChange={setLigaturesEnabled}
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div
          className="p-6 border rounded-lg bg-muted/20 transition-all"
          style={{
            fontFamily: fontFamily,
            fontSize: `${fontSize}px`,
            lineHeight: 1.4,
            fontVariantLigatures: ligaturesEnabled ? 'common-ligatures' : 'none',
            fontFeatureSettings: ligaturesEnabled ? '"liga" 1' : '"liga" 0',
          }}
        >
          {text || m.fontPreview_defaultPreviewText()}
        </div>
        {!ligaturesEnabled && (
          <p className="text-xs text-muted-foreground mt-2">
            {m.fontPreview_ligatureDisabledHint()}
          </p>
        )}
      </CardContent>
    </Card>
  );
}