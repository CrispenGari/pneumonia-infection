import { View, Text } from "react-native";
import React from "react";
import { COLORS } from "../../constants";
import { styles } from "../../styles";
import { useSettingsStore } from "../../store";

const ClassifierDate = () => {
  const {
    settings: { theme },
  } = useSettingsStore();
  return (
    <View style={{ marginVertical: 10 }}>
      <Text
        style={[
          styles.h1,
          {
            fontSize: 20,
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          },
        ]}
      >
        Pneumonia Diagnosis
      </Text>
      <Text
        style={[
          styles.p,
          {
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          },
        ]}
      >
        {new Date().toLocaleString()}
      </Text>
    </View>
  );
};

export default ClassifierDate;
