import { TouchableOpacity, ScrollView } from "react-native";
import React from "react";
import { COLORS } from "../../../../constants";
import { useSettingsStore } from "../../../../store";
import { HomeTabStacksNavProps } from "../../../../params";
import { PredictionResponse } from "../../../../types";
import AppStackBackButton from "../../../../components/AppStackBackButton/AppStackBackButton";
import { MaterialIcons } from "@expo/vector-icons";
import ClassifierDate from "../../../../components/ClassifierDate/ClassifierDate";
import { onImpact } from "../../../../utils";
import ResultImage from "../../../../components/ResultImage/ResultImage";
import BarChart from "../../../../components/BarChart/BarChart";

const Results: React.FunctionComponent<HomeTabStacksNavProps<"Results">> = ({
  navigation,
  route,
}) => {
  const results: PredictionResponse = JSON.parse(route.params.results);
  const {
    settings: { theme, ...settings },
  } = useSettingsStore();

  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerTitle: results.predictions?.top_prediction.class_label ?? "Results",
      headerRight: () => (
        <TouchableOpacity
          style={{ marginHorizontal: 20 }}
          onPress={() => {
            if (settings.haptics) {
              onImpact();
            }
            navigation.navigate("History");
          }}
          activeOpacity={0.7}
        >
          <MaterialIcons
            name="history"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        </TouchableOpacity>
      ),
      headerLeft: () => (
        <AppStackBackButton label="Home" onPress={() => navigation.goBack()} />
      ),
    });
  }, [navigation, results, settings]);

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
      <ClassifierDate />
      <ResultImage
        uri={route.params.image}
        prediction={results.predictions!.top_prediction}
      />
      <BarChart predictions={results.predictions!.all_predictions} />
    </ScrollView>
  );
};

export default Results;
