import { View } from "react-native";
import React from "react";
import DropdownSelect from "react-native-input-select";
import { COLORS, FONTS, KEYS, themes } from "../../constants";
import { SettingsType, ThemeType } from "../../types";
import { useSettingsStore } from "../../store";
import { store } from "../../utils";

const ThemeSettings = () => {
  const { settings, setSettings } = useSettingsStore();
  const changeTheme = async (theme: ThemeType) => {
    const s: SettingsType = {
      ...settings,
      theme,
    };
    await store(KEYS.APP_SETTINGS, JSON.stringify(s));
    setSettings(s);
  };
  return (
    <View style={{ marginLeft: 10 }}>
      <DropdownSelect
        placeholder="Change Theme."
        options={themes}
        optionLabel={"name"}
        optionValue={"value"}
        selectedValue={settings.theme}
        isMultiple={false}
        helperText="You can configure your theme as 'dark' or 'light'."
        dropdownContainerStyle={{
          marginBottom: 0,
          maxWidth: 300,
        }}
        dropdownIconStyle={{ top: 15, right: 15 }}
        dropdownStyle={{
          borderWidth: 0,
          paddingVertical: 8,
          paddingHorizontal: 20,
          minHeight: 40,
          maxWidth: 300,
          backgroundColor:
            settings.theme === "dark"
              ? COLORS.dark.primary
              : COLORS.light.primary,
        }}
        selectedItemStyle={{
          color:
            settings.theme === "dark"
              ? COLORS.common.white
              : COLORS.common.black,
          fontFamily: FONTS.regular,
        }}
        placeholderStyle={{
          fontFamily: FONTS.regular,
          fontSize: 18,
        }}
        onValueChange={changeTheme}
        labelStyle={{ fontFamily: FONTS.regularBold, fontSize: 20 }}
        primaryColor={
          settings.theme === "dark"
            ? COLORS.dark.tertiary
            : COLORS.light.tertiary
        }
        dropdownHelperTextStyle={{
          color:
            settings.theme === "dark"
              ? COLORS.common.white
              : COLORS.common.black,
          fontFamily: FONTS.regular,
          fontSize: 15,
        }}
        modalOptionsContainerStyle={{
          padding: 10,
          backgroundColor:
            settings.theme === "dark" ? COLORS.dark.main : COLORS.light.main,
        }}
        checkboxComponentStyles={{
          checkboxSize: 10,
          checkboxStyle: {
            backgroundColor:
              settings.theme === "dark"
                ? COLORS.dark.secondary
                : COLORS.light.secondary,
            borderRadius: 10,
            padding: 5,
            borderColor:
              settings.theme === "dark"
                ? COLORS.dark.tertiary
                : COLORS.light.tertiary,
          },
          checkboxLabelStyle: {
            color:
              settings.theme === "dark"
                ? COLORS.common.white
                : COLORS.common.black,
            fontSize: 18,
            fontFamily: FONTS.regular,
          },
        }}
      />
    </View>
  );
};

export default ThemeSettings;
