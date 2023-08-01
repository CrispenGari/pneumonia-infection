export const KEYS = {
  APP_SETTINGS: "APP_SETTINGS",
  HISTORY: "HISTORY",
  NEW_USER: "NEW_USER",
};

export const languages = [{ id: 0, name: "English", code: "en" }];

export const themes: {
  value: "dark" | "light" | "system";
  id: number;
  name: string;
}[] = [
  { id: 0, name: "Dark", value: "dark" },
  { id: 0, name: "Light", value: "light" },
  { id: 0, name: "System", value: "system" },
];

export const relativeTimeObject = {
  future: "in %s",
  past: "%s",
  s: "now",
  m: "1m",
  mm: "%dm",
  h: "1h",
  hh: "%dh",
  d: "1d",
  dd: "%dd",
  M: "1M",
  MM: "%dM",
  y: "1y",
  yy: "%dy",
};

export const models = [
  { name: "Multi Layer Perceptron (MLP)", version: "v0", id: 0 },
  { name: "LeNET", version: "v1", id: 1 },
];
export const serverBaseURL = "https://pc-djhy.onrender.com";
export const COLORS = {
  light: {
    main: "#FEF7DC",
    primary: "#E6DDC6",
    secondary: "#C2B8A3",
    tertiary: "#A19882",
  },
  dark: {
    main: "#2C3333",
    primary: "#2E4F4F",
    secondary: "#0E8388",
    tertiary: "#CBE4DE",
  },
  common: {
    gray: "gray",
    red: "#FF3953",
    rating: "#FEC700",
    url: "#00A8F5",
    white: "white",
    black: "black",
  },
};
export const Fonts = {
  YsabeauInfantItalic: require("../../assets/fonts/YsabeauInfant-Italic.ttf"),
  YsabeauInfantRegular: require("../../assets/fonts/YsabeauInfant-Regular.ttf"),
  YsabeauInfantBold: require("../../assets/fonts/YsabeauInfant-Bold.ttf"),
  YsabeauInfantBoldItalic: require("../../assets/fonts/YsabeauInfant-BoldItalic.ttf"),
  YsabeauInfantExtraBold: require("../../assets/fonts/YsabeauInfant-ExtraBold.ttf"),
  YsabeauInfantExtraBoldItalic: require("../../assets/fonts/YsabeauInfant-ExtraBoldItalic.ttf"),
};
export const FONTS = {
  regular: "YsabeauInfantRegular",
  italic: "YsabeauInfantItalic",
  italicBold: "YsabeauInfantBoldItalic",
  regularBold: "YsabeauInfantBold",
  extraBold: "YsabeauInfantExtraBold",
  extraBoldItalic: "YsabeauInfantExtraBoldItalic",
};

export const logo = require("../../assets/logo.png");

export const APP_NAME = "<APP_NAME>";
