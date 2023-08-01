import { create } from "zustand";
import { DiagnosingHistoryType, NetworkType, SettingsType } from "../types";

export const useDiagnosingHistoryStore = create<{
  diagnosingHistory: DiagnosingHistoryType[];
  setDiagnosingHistory: (diagnosingHistory: DiagnosingHistoryType[]) => void;
}>((set) => ({
  diagnosingHistory: [],
  setDiagnosingHistory: (diagnosingHistory: DiagnosingHistoryType[]) =>
    set({ diagnosingHistory }),
}));

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
  settings: {
    sound: true,
    haptics: true,
    new: true,
    theme: "light",
    historyEnabled: true,
  },
  setSettings: (settings: SettingsType) => set({ settings }),
}));
