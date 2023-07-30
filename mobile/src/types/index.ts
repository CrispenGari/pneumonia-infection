import { NetInfoStateType } from "@react-native-community/netinfo";

export type NetworkType = {
  type: NetInfoStateType | null;
  isConnected: boolean | null;
  isInternetReachable: boolean | null;
};

export type SettingsType = {
  haptics: boolean;
  sound: boolean;
  new: boolean;
};
