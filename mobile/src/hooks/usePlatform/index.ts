import { Platform } from "react-native";

export const usePlatform = () => {
  return { os: Platform.OS };
};
