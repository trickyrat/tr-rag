import { useState } from 'react';

const FALLBACK_FONTS = [
  'Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Verdana',
  'Georgia', 'Palatino', 'Garamond', 'Comic Sans MS', 'Trebuchet MS',
  'Arial Black', 'Impact', 'Lucida Grande', 'Tahoma', 'Geneva',
  'Segoe UI', 'Roboto', 'Open Sans', 'Lato', 'Montserrat'
];

export interface Font {
  family: string;
  fullName?: string;
  postscriptName?: string;
  style?: string;
}

export function useSystemFonts() {
  const [fonts, setFonts] = useState<Font[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isApiSupported, setIsApiSupported] = useState<boolean>(
    'queryLocalFonts' in window
  );
  const [isAuthorized, setIsAuthorized] = useState<boolean>(false);

  const loadFonts = async () => {
    if (!isApiSupported) {
      setError('您的浏览器不支持 Local Font Access API，已使用预设字体列表。');
      setFonts(FALLBACK_FONTS.map(family => ({ family })));
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // @ts-ignore
      const rawFonts = await window.queryLocalFonts();
      const uniqueMap = new Map<string, Font>();
      rawFonts.forEach((font: any) => {
        const family = font.family;
        if (!uniqueMap.has(family)) {
          uniqueMap.set(family, {
            family,
            fullName: font.fullName,
            postscriptName: font.postscriptName,
            style: font.style,
          });
        }
      });
      const fontList = Array.from(uniqueMap.values());
      setFonts(fontList);
      setIsAuthorized(true);
    } catch (err: any) {
      console.error('获取字体失败', err);
      if (err.name === 'NotAllowedError') {
        setError('您拒绝了字体访问权限。您可以重新加载页面并再次授权，或使用预设字体列表。');
      } else if (err.name === 'SecurityError') {
        setError('需要用户交互才能读取字体，请点击“授权并加载字体”按钮。');
      } else {
        setError('获取字体时发生错误，已使用预设字体列表。');
      }
      // Fallback to predefined fonts on error
      setFonts(FALLBACK_FONTS.map(family => ({ family })));
    } finally {
      setLoading(false);
    }
  };

  const useFallbackFonts = () => {
    setFonts(FALLBACK_FONTS.map(family => ({ family })));
    setIsAuthorized(false);
    setError(null);
  };

  return {
    fonts,
    loading,
    error,
    isApiSupported,
    isAuthorized,
    loadFonts,
    useFallbackFonts,
  };
}