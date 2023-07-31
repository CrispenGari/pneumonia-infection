import { createStackNavigator } from "@react-navigation/stack";
import { SettingsTabStacksParamList } from "../../../params";
import { COLORS, FONTS } from "../../../constants";
import {
  PrivacyPolicy,
  SettingsLanding,
  TermsOfUse,
} from "../../../screens/app/settings";
import { useSettingsStore } from "../../../store";
import { useMediaQuery } from "../../../hooks";

const Stack = createStackNavigator<SettingsTabStacksParamList>();
export const SettingsStack = () => {
  const {
    settings: { theme },
  } = useSettingsStore();
  const {
    dimension: { width },
  } = useMediaQuery();
  return (
    <Stack.Navigator
      initialRouteName="SettingsLanding"
      screenOptions={{
        headerStyle: {
          shadowOpacity: 0,
          elevation: 0,
          borderBottomColor: "transparent",
          height: width < 600 ? 100 : 80,
          backgroundColor:
            theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
        },
        headerTitleStyle: {
          fontFamily: FONTS.regularBold,
          fontSize: 24,
          color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
        },
        headerShown: true,
      }}
    >
      <Stack.Screen name="SettingsLanding" component={SettingsLanding} />
      <Stack.Screen name="PrivacyPolicy" component={PrivacyPolicy} />
      <Stack.Screen name="TermsOfUse" component={TermsOfUse} />
    </Stack.Navigator>
  );
};
