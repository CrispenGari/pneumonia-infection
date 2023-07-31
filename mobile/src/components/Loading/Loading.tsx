import { Image, Text } from "react-native";
import React from "react";
import { LinearGradient } from "expo-linear-gradient";
import { COLORS, logo } from "../../constants";
import { styles } from "../../styles";
import RippleLoadingIndicator from "../RippleLoadingIndicator/RippleLoadingIndicator";
import { useSettingsStore } from "../../store";

interface Props {
  title: string;
  withLogo: boolean;
}

const Loading: React.FunctionComponent<Props> = ({ title, withLogo }) => {
  const {
    settings: { theme },
  } = useSettingsStore();

  console.log({ theme });
  return (
    <LinearGradient
      colors={
        theme === "dark"
          ? [COLORS.dark.tertiary, COLORS.dark.main]
          : [COLORS.light.tertiary, COLORS.light.main]
      }
      start={{
        x: 0,
        y: 1,
      }}
      end={{
        x: 0,
        y: 0,
      }}
      style={{ flex: 1, justifyContent: "center", alignItems: "center" }}
    >
      {!withLogo ? (
        <RippleLoadingIndicator
          color={theme === "dark" ? COLORS.dark.main : COLORS.light.main}
          size={30}
        />
      ) : (
        <Image
          source={{
            uri: Image.resolveAssetSource(logo).uri,
          }}
          style={{
            width: 100,
            height: 100,
            marginBottom: 20,
            resizeMode: "contain",
          }}
        />
      )}
      <Text style={[styles.h1, { fontSize: 20, marginTop: 20 }]}>{title}</Text>
    </LinearGradient>
  );
};

export default Loading;
