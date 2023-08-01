import { ScrollView, Linking, Alert } from "react-native";
import React from "react";
import { SettingsTabStacksNavProps } from "../../../../params";
import { APP_NAME, COLORS, KEYS } from "../../../../constants";
import { useDiagnosingHistoryStore, useSettingsStore } from "../../../../store";
import {
  MaterialCommunityIcons,
  Entypo,
  MaterialIcons,
  Ionicons,
} from "@expo/vector-icons";
import Divider from "../../../../components/Divider/Divider";
import SettingItem from "../../../../components/SettingItem/SettingItem";
import {
  onFetchUpdateAsync,
  onImpact,
  rateApp,
  store,
} from "../../../../utils";
import { SettingsType } from "../../../../types";
import ThemeSettings from "../../../../components/ThemeSettings/ThemeSettings";
import LanguageSettings from "../../../../components/LanguageSettings/LanguageSettings";

const SettingsLanding: React.FunctionComponent<
  SettingsTabStacksNavProps<"SettingsLanding">
> = ({ navigation }) => {
  const {
    settings: { theme, ...settings },
    setSettings,
  } = useSettingsStore();
  const { diagnosingHistory, setDiagnosingHistory } =
    useDiagnosingHistoryStore();

  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerTitle: "Settings",
    });
  }, [navigation]);
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
        centered={false}
        color={theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary}
        title="MISC"
      />
      <SettingItem
        title={settings.haptics ? "Disable Haptics" : "Enable Haptics"}
        Icon={
          settings.haptics ? (
            <MaterialCommunityIcons
              name="vibrate"
              size={24}
              color={
                theme === "dark" ? COLORS.common.white : COLORS.common.black
              }
            />
          ) : (
            <MaterialCommunityIcons
              name="vibrate-off"
              size={24}
              color={
                theme === "dark" ? COLORS.common.white : COLORS.common.black
              }
            />
          )
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          const s: SettingsType = {
            ...settings,
            haptics: !settings.haptics,
            theme,
          };

          await store(KEYS.APP_SETTINGS, JSON.stringify(s));
          setSettings(s);
        }}
      />
      <SettingItem
        title={settings.sound ? "Disable App Sound" : "Enable App Sound"}
        Icon={
          settings.sound ? (
            <Ionicons
              name="volume-medium-sharp"
              size={24}
              color={
                theme === "dark" ? COLORS.common.white : COLORS.common.black
              }
            />
          ) : (
            <Ionicons
              name="volume-mute"
              size={24}
              color={
                theme === "dark" ? COLORS.common.white : COLORS.common.black
              }
            />
          )
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          const s: SettingsType = {
            ...settings,
            sound: !settings.sound,
            theme,
          };

          await store(KEYS.APP_SETTINGS, JSON.stringify(s));
          setSettings(s);
        }}
      />
      <SettingItem
        title={"Rate this Tool"}
        Icon={
          <MaterialIcons
            name="star-rate"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          await rateApp();
        }}
      />
      <SettingItem
        title={"Check for Updates"}
        Icon={
          <MaterialIcons
            name="system-update"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          await onFetchUpdateAsync();
        }}
      />
      <SettingItem
        title={"Terms of Use"}
        Icon={
          <Entypo
            name="text-document"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          navigation.navigate("TermsOfUse");
        }}
      />
      <SettingItem
        title={"Privacy Policy"}
        Icon={
          <MaterialIcons
            name="privacy-tip"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          navigation.navigate("PrivacyPolicy");
        }}
      />

      <Divider
        centered={false}
        color={theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary}
        title="DISPLAY PREFERENCES"
      />
      <ThemeSettings />
      <Divider
        centered={false}
        color={theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary}
        title="LANGUAGE PREFERENCES"
      />

      <LanguageSettings />

      <Divider
        centered={false}
        color={theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary}
        title="PNEUMONIA DIAGNOSING HISTORY"
      />

      <SettingItem
        title={
          !settings.historyEnabled
            ? "Enable Diagnosing History"
            : "Disable Diagnosing History"
        }
        Icon={
          settings.historyEnabled ? (
            <MaterialIcons
              name="lock-clock"
              size={24}
              color={
                theme === "dark" ? COLORS.common.white : COLORS.common.black
              }
            />
          ) : (
            <MaterialCommunityIcons
              name="account-clock"
              size={24}
              color={
                theme === "dark" ? COLORS.common.white : COLORS.common.black
              }
            />
          )
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          const s: SettingsType = {
            ...settings,
            theme,
            historyEnabled: !settings.historyEnabled,
          };
          await store(KEYS.APP_SETTINGS, JSON.stringify(s));
          setSettings(s);
        }}
      />
      <SettingItem
        disabled={diagnosingHistory.length === 0}
        title="Clear Diagnosing History"
        Icon={
          <MaterialIcons
            name="clear-all"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        }
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          Alert.alert(
            APP_NAME,
            `Are you sure you want to clear (${diagnosingHistory.length}) Pneumonia Diagnosis History?`,
            [
              {
                text: "Clear All",
                style: "destructive",
                onPress: async () => {
                  if (settings.haptics) {
                    onImpact();
                  }
                  await store(KEYS.DIAGNOSING_HISTORY, JSON.stringify([]));
                  setDiagnosingHistory([]);
                },
              },
              {
                text: "Cancel",
                style: "cancel",
                onPress: () => {
                  if (settings.haptics) {
                    onImpact();
                  }
                },
              },
            ],
            {
              cancelable: false,
            }
          );
        }}
      />

      <Divider
        centered={false}
        color={theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary}
        title="ISSUES & BUGS"
      />
      <SettingItem
        title="Report an Issue"
        Icon={<Entypo name="bug" size={24} color={COLORS.common.red} />}
        onPress={async () => {
          if (settings.haptics) {
            onImpact();
          }
          await Linking.openURL(
            "https://github.com/CrispenGari/pneumonia-infection/issues"
          );
        }}
      />
    </ScrollView>
  );
};

export default SettingsLanding;
