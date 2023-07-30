import "react-native-gesture-handler";
import React from "react";
import { Text, View, StatusBar, LogBox } from "react-native";
import * as Font from "expo-font";
import { Fonts } from "./src/constants";
import Routes from "./src/routes";
import ReactQueryProvider from "./src/providers/ReactQueryProvider";

LogBox.ignoreLogs;
LogBox.ignoreAllLogs();
const App = () => {
  const [ready, setReady] = React.useState<boolean>(false);
  React.useLayoutEffect(() => {
    (async () => {
      await Font.loadAsync(Fonts);
    })()
      .catch((e) => console.warn(e))
      .finally(() => setReady(true));
  }, []);

  if (!ready)
    return (
      <View style={{ flex: 1 }}>
        <Text>Loading...</Text>
      </View>
    );
  return (
    <View style={{ flex: 1 }}>
      <StatusBar barStyle={"dark-content"} />
      <ReactQueryProvider>
        <Routes />
      </ReactQueryProvider>
    </View>
  );
};
export default App;
