import { ScrollView } from "react-native";
import React from "react";
import { COLORS, KEYS } from "../../../../constants";
import { useDiagnosingHistoryStore, useSettingsStore } from "../../../../store";
import { HomeTabStacksNavProps } from "../../../../params";
import AppStackBackButton from "../../../../components/AppStackBackButton/AppStackBackButton";
import { DiagnosingHistoryType } from "../../../../types";
import { onImpact, retrieve, store } from "../../../../utils";
import Divider from "../../../../components/Divider/Divider";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import updateLocal from "dayjs/plugin/updateLocale";
import HistoryItem from "../../../../components/HistoryItem/HistoryItem";
dayjs.extend(relativeTime);
dayjs.extend(updateLocal);

const History: React.FunctionComponent<HomeTabStacksNavProps<"History">> = ({
  navigation,
  route,
}) => {
  const {
    settings: { theme, ...settings },
  } = useSettingsStore();
  const { diagnosingHistory, setDiagnosingHistory } =
    useDiagnosingHistoryStore();
  React.useEffect(() => {
    (async () => {
      const h = await retrieve(KEYS.DIAGNOSING_HISTORY);
      const hh: DiagnosingHistoryType[] = h ? JSON.parse(h) : diagnosingHistory;
      setDiagnosingHistory(hh);
    })();
  }, []);
  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerLeft: () => (
        <AppStackBackButton
          label={route.params.from}
          onPress={() => {
            if (settings.haptics) {
              onImpact();
            }
            navigation.goBack();
          }}
        />
      ),
    });
  }, [navigation, route]);

  const deleteHistoryItem = async (id: string) => {
    const histories = diagnosingHistory.filter((h) => h.id !== id);
    await store(KEYS.DIAGNOSING_HISTORY, JSON.stringify(histories));
    setDiagnosingHistory(histories);
  };

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
      contentContainerStyle={{ paddingBottom: 140 }}
    >
      <Divider
        centered={true}
        title="LAST 24 HOURS"
        color={COLORS.light.secondary}
      />

      {diagnosingHistory
        .map((hist) => {
          const date1 = dayjs();
          const date2 = dayjs();
          const hours = date2.diff(date1, "hours");
          const days = Math.floor(hours / 24);
          return { ...hist, days };
        })
        .filter((hist) => hist.days === 0)
        .map((hist) => (
          <HistoryItem
            key={hist.id}
            navigation={navigation}
            hist={hist}
            deleteHistoryItem={deleteHistoryItem}
          />
        ))}

      <Divider centered={true} title="OLD" color={COLORS.light.secondary} />
      {diagnosingHistory
        .map((hist) => {
          const date1 = dayjs();
          const date2 = dayjs();
          const hours = date2.diff(date1, "hours");
          const days = Math.floor(hours / 24);
          return { ...hist, days };
        })
        .filter((hist) => hist.days > 0)
        .map((hist) => (
          <HistoryItem
            key={hist.id}
            navigation={navigation}
            hist={hist}
            deleteHistoryItem={deleteHistoryItem}
          />
        ))}
    </ScrollView>
  );
};

export default History;
