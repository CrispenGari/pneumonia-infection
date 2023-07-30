import { COLORS } from "../../constants";
import { AppParamList } from "../../params";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { HomeStack } from "./home";
import { SettingsStack } from "./settings";
import TabIcon from "../../components/TabIcon/TabIcon";
import { MaterialCommunityIcons, Ionicons } from "@expo/vector-icons";

import React from "react";
const Tab = createBottomTabNavigator<AppParamList>();

export const AppTabs = () => {
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
          backgroundColor: COLORS.dark.primary,
          paddingVertical: 10,
          height: 80,
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
                name: "home-account",
                IconComponent: MaterialCommunityIcons,
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
                name: "settings",
                IconComponent: Ionicons,
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
