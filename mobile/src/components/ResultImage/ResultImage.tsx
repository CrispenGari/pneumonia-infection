import { View, Image } from "react-native";
import React from "react";
import { useSettingsStore } from "../../store";
import { useMediaQuery } from "../../hooks";
import { PredictionType } from "../../types";
import { COLORS } from "../../constants";
import TypeWriter from "react-native-typewriter";
import { styles } from "../../styles";

interface Props {
  prediction: PredictionType;
  uri: string;
}

const ResultImage: React.FunctionComponent<Props> = ({ prediction, uri }) => {
  const {
    settings: { theme },
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
      <TypeWriter
        style={[
          styles.h1,
          {
            fontSize: 20,
            marginBottom: 20,
            color:
              prediction.class_label.toLowerCase() !== "normal"
                ? COLORS.common.red
                : theme === "dark"
                ? COLORS.common.white
                : COLORS.common.black,
          },
        ]}
        typing={1}
        maxDelay={-50}
      >
        {prediction.class_label.toLowerCase()}
      </TypeWriter>
      <Image
        source={{
          uri,
        }}
        style={{
          width: "100%",
          height: "70%",
          resizeMode: "cover",
          borderRadius: 10,
          flex: 1,
        }}
      />
      <TypeWriter
        style={[
          styles.h1,
          {
            fontSize: 20,
            marginVertical: 10,
            color:
              prediction.probability < 0.6
                ? COLORS.common.red
                : theme === "dark"
                ? COLORS.common.white
                : COLORS.common.black,
          },
        ]}
        typing={1}
        maxDelay={-50}
      >
        The model is {(prediction.probability * 100).toFixed(1)}% confident that
        the chest X-Ray provided for diagnosis belongs to{" "}
        {prediction.class_label.toLowerCase()}.
      </TypeWriter>
    </View>
  );
};

export default ResultImage;
