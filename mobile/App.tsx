import "react-native-gesture-handler";
import React from "react";
import { View, LogBox, StatusBar } from "react-native";
import * as Font from "expo-font";
import { Fonts, KEYS } from "./src/constants";
import Routes from "./src/routes";
import ReactQueryProvider from "./src/providers/ReactQueryProvider";
import Loading from "./src/components/Loading/Loading";
import { useSettingsStore } from "./src/store";
import { SettingsType } from "./src/types";
import { retrieve } from "./src/utils";

LogBox.ignoreLogs;
LogBox.ignoreAllLogs();
const App = () => {
  const [ready] = Font.useFonts(Fonts);
  const {
    setSettings,
    settings: { theme },
  } = useSettingsStore();
  React.useEffect(() => {
    (async () => {
      const s = await retrieve(KEYS.APP_SETTINGS);
      if (s) {
        const ss: SettingsType = await JSON.parse(s);
        setSettings(ss);
      }
    })();
  }, []);
  if (!ready)
    return <Loading fontReady={ready} withLogo={true} title="Loading..." />;
  return (
    <View style={{ flex: 1 }}>
      <StatusBar
        barStyle={theme === "light" ? "dark-content" : "light-content"}
      />
      <ReactQueryProvider>
        <Routes />
      </ReactQueryProvider>
    </View>
  );
};
export default App;
