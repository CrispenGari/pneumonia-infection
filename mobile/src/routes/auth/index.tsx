import { createStackNavigator } from "@react-navigation/stack";
import { Landing } from "../../screens/auth";
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
    </Stack.Navigator>
  );
};
