import { createStackNavigator } from "@react-navigation/stack";
import { COLORS, FONTS } from "../../../constants";
import { Classifier, History, Results } from "../../../screens/app/home";
import { HomeTabStacksParamList } from "../../../params";
import { useSettingsStore } from "../../../store";
import { TouchableOpacity } from "react-native-gesture-handler";
import { MaterialIcons } from "@expo/vector-icons";

const Stack = createStackNavigator<HomeTabStacksParamList>();

export const HomeStack = () => {
  const {
    settings: { theme },
  } = useSettingsStore();
  return (
    <Stack.Navigator
      initialRouteName="Classifier"
      screenOptions={{
        headerStyle: {
          shadowOpacity: 0,
          elevation: 0,
          borderBottomColor: "transparent",
          height: 80,
          backgroundColor:
            theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
        },
        headerTitleStyle: {
          fontFamily: FONTS.regularBold,
          fontSize: 24,
          color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
        },
        headerShown: true,
        headerRight: (props) => (
          <TouchableOpacity
            style={{ marginHorizontal: 20 }}
            onPress={() => props}
            activeOpacity={0.7}
          >
            <MaterialIcons
              name="history"
              size={24}
              color={
                theme === "dark" ? COLORS.common.white : COLORS.common.black
              }
            />
          </TouchableOpacity>
        ),
      }}
    >
      <Stack.Screen name="Classifier" component={Classifier} />
      <Stack.Screen name="History" component={History} />
      <Stack.Screen name="Results" component={Results} />
    </Stack.Navigator>
  );
};
