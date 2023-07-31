import { View, Text, TouchableOpacity, Image } from "react-native";
import React from "react";
import { Swipeable } from "react-native-gesture-handler";
import { PredictionResponse } from "../../types";
import { onImpact } from "../../utils";
import { styles } from "../../styles";
import { MaterialIcons } from "@expo/vector-icons";
import { COLORS, relativeTimeObject } from "../../constants";
import { useSettingsStore } from "../../store";

import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import updateLocal from "dayjs/plugin/updateLocale";
import { HomeTabStacksParamList } from "../../params";
import { StackNavigationProp } from "@react-navigation/stack";
dayjs.extend(relativeTime);
dayjs.extend(updateLocal);
dayjs.updateLocale("en", {
  relativeTime: relativeTimeObject,
});
interface Props {
  hist: {
    days: number;
    date: Date;
    result: PredictionResponse;
    image: string;
    id: string;
  };
  deleteHistoryItem: (id: string) => Promise<void>;
  navigation: StackNavigationProp<HomeTabStacksParamList, "History">;
}
const HistoryItem: React.FunctionComponent<Props> = ({
  hist: { image, id, date, result },
  deleteHistoryItem,
  navigation,
}) => {
  const swipeableRef = React.useRef<Swipeable | undefined>();
  const {
    settings: { theme, ...settings },
  } = useSettingsStore();

  return (
    <Swipeable
      ref={swipeableRef as any}
      renderRightActions={(_progress, _dragX) => {
        return (
          <TouchableOpacity
            activeOpacity={0.7}
            style={{
              justifyContent: "center",
              alignItems: "center",
              minWidth: 50,
              backgroundColor: COLORS.common.red,
              borderTopLeftRadius: 0,
              borderBottomLeftRadius: 0,
            }}
            onPress={() => {
              if (settings.haptics) {
                onImpact();
              }
              deleteHistoryItem(id);
            }}
          >
            <MaterialIcons name="delete" size={24} color="white" />
          </TouchableOpacity>
        );
      }}
    >
      <TouchableOpacity
        style={{
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          padding: 5,
          flex: 1,
          paddingHorizontal: 10,
        }}
        activeOpacity={0.7}
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          navigation.navigate("Results", {
            image,
            results: JSON.stringify(result),
            from: "History",
          });
        }}
      >
        <Image
          source={{ uri: image }}
          style={{
            height: 40,
            width: 40,
            objectFit: "cover",
            marginRight: 10,
            borderRadius: 3,
          }}
        />
        <View style={{ flex: 1, flexDirection: "row" }}>
          <View style={{ flex: 1 }}>
            <Text
              style={[
                styles.h1,
                {
                  color:
                    result.predictions?.top_prediction.class_label.toLowerCase() !==
                    "normal"
                      ? COLORS.common.red
                      : theme === "dark"
                      ? COLORS.common.white
                      : COLORS.common.black,
                },
              ]}
            >
              {result.predictions?.top_prediction?.class_label}
            </Text>
            <Text
              style={[styles.p, { fontSize: 14, color: COLORS.common.gray }]}
            >
              {(result.predictions!.top_prediction.probability * 100).toFixed(
                0
              )}
              % • {result.predictions?.top_prediction.class_label} •{" "}
              {dayjs(date).fromNow()}
            </Text>
          </View>
        </View>

        <MaterialIcons
          name="history"
          size={24}
          color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
        />
      </TouchableOpacity>
    </Swipeable>
  );
};

export default HistoryItem;
