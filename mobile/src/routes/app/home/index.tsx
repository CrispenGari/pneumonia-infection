import { createStackNavigator } from "@react-navigation/stack";
import { COLORS, FONTS } from "../../../constants";
import { Classifier, History } from "../../../screens/app/home";
import { HomeTabStacksParamList } from "../../../params";

const Stack = createStackNavigator<HomeTabStacksParamList>();

export const HomeStack = () => {
  return (
    <Stack.Navigator
      initialRouteName="Classifier"
      screenOptions={{
        headerStyle: {
          shadowOpacity: 0,
          elevation: 0,
          borderBottomColor: "transparent",
          height: 100,
          backgroundColor: COLORS.dark.main,
        },
        headerTitleStyle: {
          fontFamily: FONTS.regularBold,
          fontSize: 24,
        },
        headerShown: true,
      }}
    >
      <Stack.Screen name="Classifier" component={Classifier} />
      <Stack.Screen name="History" component={History} />
    </Stack.Navigator>
  );
};
