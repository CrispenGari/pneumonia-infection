import { NetInfoStateType } from "@react-native-community/netinfo";
import { type ColorSchemeName } from "react-native";

export type NetworkType = {
  type: NetInfoStateType | null;
  isConnected: boolean | null;
  isInternetReachable: boolean | null;
};

export type SettingsType = {
  haptics: boolean;
  sound: boolean;
  new: boolean;
  theme: ThemeType;
};

export type ThemeType = ColorSchemeName;
