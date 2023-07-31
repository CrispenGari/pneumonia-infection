import { NetInfoStateType } from "@react-native-community/netinfo";
import { type ColorSchemeName } from "react-native";

export type NetworkType = {
  type: NetInfoStateType | null;
  isConnected: boolean | null;
  isInternetReachable: boolean | null;
};

export type ThemeType = ColorSchemeName;
export type SettingsType = {
  haptics: boolean;
  sound: boolean;
  new: boolean;
  theme: ThemeType;
};
export interface PredictionType {
  class_label: string;
  label: number;
  probability: number;
}
export type MetaType = {
  description: string;
  language: string;
  library: string;
  main: string;
  programmer: string;
};
export type PredictionResponse = {
  modelVersion: "v0" | "v1";
  success: boolean;
  prediction?: {
    top_prediction: PredictionType;
    all_predictions: PredictionType[];
  };
  meta: MetaType;
};
