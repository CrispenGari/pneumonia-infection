import { View, Text } from "react-native";
import React from "react";
import { BarChart } from "react-native-chart-kit";
import { COLORS } from "../../constants";
import { PredictionType } from "../../types";
import { useSettingsStore } from "../../store";
import { styles } from "../../styles";
interface Props {
  predictions: PredictionType[];
}
const Bar: React.FunctionComponent<Props> = ({ predictions }) => {
  const [labels, setLabels] = React.useState<string[]>([]);
  const [data, setData] = React.useState<number[]>([]);
  const {
    settings: { theme },
  } = useSettingsStore();

  React.useEffect(() => {
    let mounted = true;
    if (mounted) {
      setData(predictions.map((pred) => Math.ceil(pred.probability * 100)));
      setLabels(
        predictions.map((pred) =>
          pred.class_label.toLowerCase().replace("pneumonia", "")
        )
      );
    }
    return () => {
      mounted = false;
    };
  }, [predictions]);

  return (
    <View
      style={{
        width: "100%",
        alignSelf: "center",
        maxWidth: 400,
        marginVertical: 20,
      }}
    >
      <Text
        style={[
          styles.h1,
          {
            fontSize: 20,
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
            letterSpacing: 1,
          },
        ]}
      >
        Diagnostic Distribution
      </Text>
      <BarChart
        style={{
          marginTop: 10,
          width: "100%",
          justifyContent: "center",
          borderRadius: 10,
        }}
        data={{
          labels,
          datasets: [
            {
              data,
              withDots: true,
              color: (_opacity = 1) => COLORS.common.red,
              withScrollableDot: true,
            },
          ],
        }}
        fromZero
        width={400}
        height={400}
        yAxisLabel="  "
        yAxisSuffix=""
        showBarTops={false}
        chartConfig={{
          backgroundGradientFrom: COLORS.common.white,
          backgroundGradientFromOpacity: 1,
          backgroundGradientTo:
            theme === "dark" ? COLORS.dark.secondary : COLORS.light.secondary,
          backgroundGradientToOpacity: 0.5,
          paddingRight: 0,
          horizontalOffset: 0,
          color: (opacity = 1) => `rgba(57, 91, 100, ${opacity})`,
          strokeWidth: 1,
          barPercentage: 1,
          propsForDots: {
            r: "2",
            strokeWidth: "2",
            stroke: COLORS.common.white,
          },
          fillShadowGradient: COLORS.dark.main,
          fillShadowGradientOpacity: 1,
          fillShadowGradientTo: COLORS.common.red,
          scrollableDotFill: "%",
          formatTopBarValue: (val) => `${val.toFixed(0)}%`,
          formatYLabel: (val) => `${Number(val).toFixed(0)}%`,
        }}
        verticalLabelRotation={60}
        horizontalLabelRotation={0}
        showValuesOnTopOfBars
        yLabelsOffset={10}
      />
    </View>
  );
};

export default Bar;
