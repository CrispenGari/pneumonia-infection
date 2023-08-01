import { Text, TouchableOpacity } from "react-native";
import React from "react";
import { styles } from "../../styles";
import { COLORS } from "../../constants";
import { useSettingsStore } from "../../store";

interface Props {
  onPress: () => void;
  title: string;
  Icon: React.ReactNode;
  disabled?: boolean;
}
const SettingItem: React.FunctionComponent<Props> = ({
  onPress,
  title,
  Icon,
  disabled,
}) => {
  const {
    settings: { theme },
  } = useSettingsStore();
  return (
    <TouchableOpacity
      disabled={disabled}
      style={{
        flexDirection: "row",
        justifyContent: "space-between",
        paddingHorizontal: 10,
        padding: 10,
        marginBottom: 1,
        backgroundColor:
          theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
        alignItems: "center",
      }}
      onPress={onPress}
      activeOpacity={0.7}
    >
      {Icon}
      <Text
        style={[
          styles.p,
          {
            flex: 1,
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
            marginLeft: 10,
          },
        ]}
      >
        {title}
      </Text>
    </TouchableOpacity>
  );
};

export default SettingItem;
