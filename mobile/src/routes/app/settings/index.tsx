import { createStackNavigator } from "@react-navigation/stack";
import { SettingsTabStacksParamList } from "../../../params";
import { COLORS, FONTS } from "../../../constants";
import {
  PrivacyPolicy,
  SettingsLanding,
  TermsOfUse,
} from "../../../screens/app/settings";

const Stack = createStackNavigator<SettingsTabStacksParamList>();

export const SettingsStack = () => {
  return (
    <Stack.Navigator
      initialRouteName="SettingsLanding"
      screenOptions={{
        headerStyle: {
          shadowOpacity: 0,
          elevation: 0,
          borderBottomColor: "transparent",
          height: 100,
          backgroundColor: COLORS.common.rating,
        },
        headerTitleStyle: {
          fontFamily: FONTS.regularBold,
          fontSize: 24,
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
