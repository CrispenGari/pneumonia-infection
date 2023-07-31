import { TouchableOpacity, ScrollView } from "react-native";
import React from "react";
import { COLORS, models } from "../../../../constants";
import { useSettingsStore } from "../../../../store";
import { HomeTabStacksNavProps } from "../../../../params";
import { PredictionResponse } from "../../../../types";
import AppStackBackButton from "../../../../components/AppStackBackButton/AppStackBackButton";
import { MaterialIcons } from "@expo/vector-icons";
import ClassifierDate from "../../../../components/ClassifierDate/ClassifierDate";
import { onImpact } from "../../../../utils";
import ResultImage from "../../../../components/ResultImage/ResultImage";
import BarChart from "../../../../components/BarChart/BarChart";
import TableComponent from "../../../../components/TableComponent/TableComponent";

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
      headerTitle:
        results.predictions?.top_prediction.class_label
          .replace("Pneumonia".toUpperCase(), "")
          .trim() ?? "Results",
      headerRight: () => (
        <TouchableOpacity
          style={{ marginHorizontal: 20 }}
          onPress={() => {
            if (settings.haptics) {
              onImpact();
            }
            navigation.navigate("History", {
              from: "Results",
            });
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
        <AppStackBackButton
          label={route.params.from}
          onPress={() => navigation.goBack()}
        />
      ),
    });
  }, [navigation, results, settings, route]);

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

      <TableComponent
        title="Possible Pneumonia Outcomes"
        tableHead={["Class Name", "Probability", "Class Label"]}
        tableData={[
          ...results.predictions!.all_predictions.map(
            ({ class_label, label, probability }) => [
              class_label.toLowerCase().replace("pneumonia", ""),
              probability.toFixed(2).toString(),
              label.toString(),
            ]
          ),
        ]}
      />

      <BarChart predictions={results.predictions!.all_predictions} />
      <TableComponent
        title="Model Version"
        tableHead={["Model Name", "Model Version"]}
        tableData={[
          [
            models.find((model) => model.version === results.modelVersion)!
              .name,
            results.modelVersion,
          ],
        ]}
      />
    </ScrollView>
  );
};

export default Results;
