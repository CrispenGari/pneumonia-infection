import "react-native-gesture-handler";
import React from "react";
import { View, StatusBar, LogBox } from "react-native";
import * as Font from "expo-font";
import { Fonts, KEYS } from "./src/constants";
import Routes from "./src/routes";
import ReactQueryProvider from "./src/providers/ReactQueryProvider";
import Loading from "./src/components/Loading/Loading";
import { Appearance, useColorScheme } from "react-native";
import { useSettingsStore } from "./src/store";
import { SettingsType } from "./src/types";
import { retrieve, store } from "./src/utils";

LogBox.ignoreLogs;
LogBox.ignoreAllLogs();
const App = () => {
  const [ready] = Font.useFonts(Fonts);
  const theme = useColorScheme();
  const { settings, setSettings } = useSettingsStore();
  Appearance.addChangeListener(async ({ colorScheme }) => {
    const s: SettingsType = {
      ...settings,
      theme: colorScheme,
    };
    await store(KEYS.APP_SETTINGS, JSON.stringify(s));
    setSettings(s);
  });

  React.useEffect(() => {
    (async () => {
      const s: SettingsType = {
        ...settings,
        theme,
      };
      await store(KEYS.APP_SETTINGS, JSON.stringify(s));
      setSettings(s);
    })();
  }, [theme]);

  React.useEffect(() => {
    (async () => {
      const ss = await retrieve(KEYS.APP_SETTINGS);
      if (ss) {
        const s: SettingsType = JSON.parse(ss);
        setSettings(s);
      }
    })();
  }, [theme]);
  if (!ready) return <Loading withLogo={true} title="Loading..." />;
  return (
    <View style={{ flex: 1 }}>
      <StatusBar
        barStyle={settings.theme === "light" ? "dark-content" : "light-content"}
      />
      <ReactQueryProvider>
        <Routes />
      </ReactQueryProvider>
    </View>
  );
};
export default App;
