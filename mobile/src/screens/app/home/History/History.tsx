import { View, Text, ScrollView } from "react-native";
import React from "react";
import { COLORS } from "../../../../constants";
import { useSettingsStore } from "../../../../store";

const History = () => {
  const {
    settings: { theme, ...settings },
  } = useSettingsStore();
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
      contentContainerStyle={{ padding: 10, paddingBottom: 140 }}
    >
      <Text>History</Text>
    </ScrollView>
  );
};

export default History;
