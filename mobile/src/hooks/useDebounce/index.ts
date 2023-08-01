import React from "react";

export const useDebounce = <T>(value: T, delay: number): T => {
  const [v, setV] = React.useState<T>(value);
  React.useEffect(() => {
    const timeOutId = setTimeout(() => {
      setV(value);
    }, delay);
    return () => {
      clearTimeout(timeOutId);
    };
  }, [delay, value]);
  return v;
};
