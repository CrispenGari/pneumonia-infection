import { RouteProp } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { BottomTabNavigationProp } from "@react-navigation/bottom-tabs";

export type AuthParamList = {
  Landing: undefined;
  AuthPrivacyPolicy: {
    from: keyof AuthParamList;
  };
  AuthTermsOfUse: { from: keyof AuthParamList };
};

export type AuthNavProps<T extends keyof AuthParamList> = {
  navigation: StackNavigationProp<AuthParamList, T>;
  route: RouteProp<AuthParamList, T>;
};

// Tabs
export type AppParamList = {
  Home: undefined;
  Settings: undefined;
};

export type AppNavProps<T extends keyof AppParamList> = {
  navigation: BottomTabNavigationProp<AppParamList, T>;
  route: RouteProp<AppParamList, T>;
};

// Home Tab Stacks

export type HomeTabStacksParamList = {
  Classifier: undefined;
  History: {
    from: "Home" | "Results";
  };
  Results: {
    results: string;
    image: string;
    from: "Home" | "History";
  };
};

export type HomeTabStacksNavProps<T extends keyof HomeTabStacksParamList> = {
  navigation: StackNavigationProp<HomeTabStacksParamList, T>;
  route: RouteProp<HomeTabStacksParamList, T>;
};

// Settings TabStacks

export type SettingsTabStacksParamList = {
  SettingsLanding: undefined;
  PrivacyPolicy: {
    from: keyof SettingsTabStacksParamList;
  };
  TermsOfUse: { from: keyof SettingsTabStacksParamList };
};

export type SettingsTabStacksNavProps<
  T extends keyof SettingsTabStacksParamList
> = {
  navigation: StackNavigationProp<SettingsTabStacksParamList, T>;
  route: RouteProp<SettingsTabStacksParamList, T>;
};
