import { NavigationContainer } from "@react-navigation/native";
import * as Linking from "expo-linking";
import React from "react";
import { AuthStack } from "./auth";
import NetInfo from "@react-native-community/netinfo";
import { useNetworkStore } from "../store";
import { AppTabs } from "./app";
import { APP_NAME } from "../constants";

const Routes = () => {
  const prefix = Linking.createURL("/");
  const { setNetwork } = useNetworkStore();

  React.useEffect(() => {
    const unsubscribe = NetInfo.addEventListener(
      ({ type, isInternetReachable, isConnected }) => {
        setNetwork({ type, isConnected, isInternetReachable });
      }
    );
    return () => unsubscribe();
  }, [setNetwork]);

  return (
    <NavigationContainer
      linking={{
        prefixes: [
          prefix,
          `${APP_NAME}://`,
          `https://${APP_NAME}.com`,
          `https://*.invitee.com`,
        ],
        config: {
          screens: {
            Home: "home",
            Settings: "settings",
          },
        },
      }}
    >
      {false ? <AppTabs /> : <AuthStack />}
    </NavigationContainer>
  );
};

export default Routes;
