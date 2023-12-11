import { createStackNavigator } from "@react-navigation/stack";
import { Landing, AuthPrivacyPolicy, AuthTermsOfUse } from "../../screens/auth";
import { AuthParamList } from "../../params";

const Stack = createStackNavigator<AuthParamList>();
export const AuthStack = () => {
  return (
    <Stack.Navigator
      initialRouteName="Landing"
      screenOptions={{
        headerShown: false,
      }}
    >
      <Stack.Screen name="Landing" component={Landing} />
      <Stack.Screen name="AuthPrivacyPolicy" component={AuthPrivacyPolicy} />
      <Stack.Screen name="AuthTermsOfUse" component={AuthTermsOfUse} />
    </Stack.Navigator>
  );
};
