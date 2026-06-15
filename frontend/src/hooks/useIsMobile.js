// frontend/src/hooks/useIsMobile.js
import { useState, useEffect } from "react";

function useWindowWidth() {
  const [w, setW] = useState(() => window?.innerWidth ?? 1024);
  useEffect(() => {
    const fn = () => setW(window.innerWidth);
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, []);
  return w;
}

export const useIsMobile = () => useWindowWidth() <= 640;
export const useIsTablet = () => useWindowWidth() <= 1024;

// Responsive grid style helper
export function gridCols(desktop, tablet, mobile, isMobile, isTablet) {
  if (isMobile) return `repeat(${mobile}, 1fr)`;
  if (isTablet) return `repeat(${tablet}, 1fr)`;
  return `repeat(${desktop}, 1fr)`;
}