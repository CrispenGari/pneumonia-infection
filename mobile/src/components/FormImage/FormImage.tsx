import { View, Text, TouchableOpacity, Image } from "react-native";
import React from "react";
import { COLORS, logo } from "../../constants";
import { styles } from "../../styles";
import { onImpact } from "../../utils";
import { useMediaQuery } from "../../hooks";
import { useSettingsStore } from "../../store";
import * as Animatable from "react-native-animatable";
interface Props {
  image: {
    uri: string;
    name: string;
  } | null;
  setImage: React.Dispatch<
    React.SetStateAction<{
      uri: string;
      name: string;
    } | null>
  >;
  diagnosing: boolean;
}
const FormImage: React.FunctionComponent<Props> = ({
  image,
  setImage,
  diagnosing,
}) => {
  const {
    settings: { theme, ...settings },
  } = useSettingsStore();
  const {
    dimension: { width },
  } = useMediaQuery();
  return (
    <View
      style={{
        alignSelf: "center",
        width: width < 600 ? 300 : 400,
        height: width < 600 ? 300 : 400,
        backgroundColor:
          theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
        borderRadius: 10,
        justifyContent: "center",
        alignItems: "center",
        padding: 10,
      }}
    >
      {image ? (
        <>
          <Text
            style={[
              styles.h1,
              {
                fontSize: 20,
                marginVertical: 5,
                color:
                  theme === "dark" ? COLORS.common.white : COLORS.common.black,
              },
            ]}
          >
            Selected X-ray Image
          </Text>
          <Image
            source={{
              uri: image.uri,
            }}
            style={{
              width: "100%",
              height: "70%",
              resizeMode: "cover",
              borderRadius: 10,
              flex: 1,
            }}
          />
          <TouchableOpacity
            activeOpacity={0.7}
            disabled={diagnosing}
            onPress={() => {
              if (settings.haptics) {
                onImpact();
              }
              setImage(null);
            }}
            style={[
              styles.button,
              {
                backgroundColor: COLORS.common.red,
                marginTop: 10,
                borderRadius: 5,
                padding: 7,
                maxWidth: "100%",
              },
            ]}
          >
            <Text
              style={[
                styles.button__text,
                {
                  color: COLORS.common.white,
                },
              ]}
            >
              REMOVE
            </Text>
          </TouchableOpacity>
        </>
      ) : (
        <>
          <Animatable.Image
            animation={"bounce"}
            duration={2000}
            iterationCount={"infinite"}
            easing={"linear"}
            direction={"normal"}
            useNativeDriver={false}
            source={logo}
            style={{
              width: 100,
              height: 100,
              marginVertical: 30,
              resizeMode: "contain",
              tintColor: COLORS.common.white,
            }}
          />
          <Text
            style={[
              styles.h1,
              {
                fontSize: 20,
                color:
                  theme === "dark" ? COLORS.common.white : COLORS.common.black,
              },
            ]}
          >
            Select a chest X-ray Image
          </Text>
        </>
      )}
    </View>
  );
};

export default FormImage;
