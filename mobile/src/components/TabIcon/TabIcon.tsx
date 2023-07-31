import React from "react";
import { View, Text } from "react-native";
import { COLORS } from "../../constants";
import { useMediaQuery } from "../../hooks";
import { useSettingsStore } from "../../store";
interface IconI {
  IconComponent: any;
  name: string;
}
interface Props {
  title?: string;
  Icon: IconI;
  focused: boolean;
}
const TabIcon: React.FC<Props> = ({ focused, Icon, title }) => {
  const {
    dimension: { width },
  } = useMediaQuery();

  const {
    settings: { theme },
  } = useSettingsStore();
  return (
    <View
      style={[
        {
          flex: 1,
          justifyContent: "center",
          alignItems: "center",
          position: "relative",
          width: width >= 600 ? 300 : 100,
        },
      ]}
    >
      {theme === "dark" ? (
        <>
          <Icon.IconComponent
            name={Icon.name}
            size={20}
            color={!focused ? COLORS.common.white : COLORS.dark.secondary}
          />
          <Text
            style={{
              color: !focused ? COLORS.common.white : COLORS.dark.secondary,
            }}
          >
            {title}
          </Text>
        </>
      ) : (
        <>
          <Icon.IconComponent
            name={Icon.name}
            size={20}
            color={focused ? COLORS.common.black : COLORS.light.tertiary}
          />
          <Text
            style={{
              color: focused ? COLORS.common.black : COLORS.light.tertiary,
            }}
          >
            {title}
          </Text>
        </>
      )}
    </View>
  );
};

export default TabIcon;
