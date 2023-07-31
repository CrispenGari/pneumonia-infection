import { COLORS } from "../../constants";
import { AppParamList } from "../../params";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { HomeStack } from "./home";
import { SettingsStack } from "./settings";
import TabIcon from "../../components/TabIcon/TabIcon";
import { AntDesign } from "@expo/vector-icons";

import React from "react";
import { useSettingsStore } from "../../store";
import { useMediaQuery } from "../../hooks";
const Tab = createBottomTabNavigator<AppParamList>();

export const AppTabs = () => {
  const {
    settings: { theme },
  } = useSettingsStore();
  const {
    dimension: { width },
  } = useMediaQuery();
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
