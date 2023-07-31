import { View, Text, ScrollView } from "react-native";
import React from "react";
import { SettingsTabStacksNavProps } from "../../../../params";
import { COLORS } from "../../../../constants";
import { useSettingsStore } from "../../../../store";

const SettingsLanding: React.FunctionComponent<
  SettingsTabStacksNavProps<"SettingsLanding">
> = ({ navigation }) => {
  const {
    settings: { theme },
  } = useSettingsStore();
  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerTitle: "Settings",
    });
  }, [navigation]);
  return (
    <ScrollView
      scrollEventThrottle={16}
      showsHorizontalScrollIndicator={false}
      showsVerticalScrollIndicator={false}
      style={{
        flex: 1,
        backgroundColor:
          theme === "dark" ? COLORS.dark.main : COLORS.light.main,
      }}
      contentContainerStyle={{ paddingBottom: 140 }}
    >
      <Text>SettingsLanding</Text>
    </ScrollView>
  );
};

export default SettingsLanding;
