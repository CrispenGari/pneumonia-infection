import { create } from "zustand";
import { NetworkType, SettingsType } from "../types";

export const useNetworkStore = create<{
  network: Required<NetworkType>;
  setNetwork: (network: Required<NetworkType>) => void;
}>((set) => ({
  network: {
    isConnected: true,
    isInternetReachable: true,
    type: null,
  },
  setNetwork: (network: Required<NetworkType>) => set({ network }),
}));

export const useSettingsStore = create<{
  settings: Required<SettingsType>;
  setSettings: (settings: SettingsType) => void;
}>((set) => ({
  settings: { sound: true, haptics: true, new: false },
  setSettings: (settings: SettingsType) => set({ settings }),
}));
