import { APP_NAME, COLORS } from "../../constants";
import { AppParamList } from "../../params";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { HomeStack } from "./home";
import { Alert } from "react-native";
import { SettingsStack } from "./settings";
import TabIcon from "../../components/TabIcon/TabIcon";
import { AntDesign } from "@expo/vector-icons";
import React from "react";
import { useNetworkStore, useSettingsStore } from "../../store";
import { useMediaQuery } from "../../hooks";
import { onImpact, onNotification } from "../../utils";
const Tab = createBottomTabNavigator<AppParamList>();

export const AppTabs = () => {
  const {
    settings: { theme, haptics },
  } = useSettingsStore();
  const {
    dimension: { width },
  } = useMediaQuery();
  const { network } = useNetworkStore();

  React.useEffect(() => {
    if (!network.isInternetReachable && network.isInternetReachable !== null) {
      if (haptics) {
        onNotification();
      }
      Alert.alert(
        APP_NAME,
        "We have detected that you don't have active internet connection, diagnosing pneumonia requires a stable internet connection, you can view your diagnostic history.",
        [
          {
            text: "CANCEL",
            style: "destructive",
            onPress: () => {
              if (haptics) {
                onImpact();
              }
            },
          },
        ],
        {
          cancelable: false,
        }
      );
    }
  }, [network, haptics]);
  return (
    <Tab.Navigator
      initialRouteName="Home"
      screenOptions={{
        headerShown: false,
        tabBarHideOnKeyboard: true,
        tabBarStyle: {
          elevation: 0,
          shadowOpacity: 0,
          borderTopWidth: 0,
          borderColor: "transparent",
          backgroundColor:
            theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
          paddingVertical: 10,
          height: width < 600 ? 80 : 60,
          width: "auto",
        },
        tabBarShowLabel: false,
        tabBarBadgeStyle: {
          backgroundColor: "cornflowerblue",
          color: "white",
          fontSize: 10,
          maxHeight: 20,
          maxWidth: 20,
          marginLeft: 3,
        },
        tabBarVisibilityAnimationConfig: {
          hide: {
            animation: "timing",
          },
          show: {
            animation: "spring",
          },
        },
        tabBarItemStyle: {
          width: "auto",
        },
      }}
    >
      <Tab.Screen
        options={{
          tabBarIcon: (props) => (
            <TabIcon
              {...props}
              title="home"
              Icon={{
                name: "home",
                IconComponent: AntDesign,
              }}
            />
          ),
        }}
        name="Home"
        component={HomeStack}
      />

      <Tab.Screen
        options={{
          tabBarIcon: (props) => (
            <TabIcon
              {...props}
              title="settings"
              Icon={{
                name: "setting",
                IconComponent: AntDesign,
              }}
            />
          ),
        }}
        name="Settings"
        component={SettingsStack}
      />
    </Tab.Navigator>
  );
};
