import { View } from "react-native";
import React from "react";
import DropdownSelect from "react-native-input-select";
import { COLORS, FONTS, languages } from "../../constants";
import { useSettingsStore } from "../../store";

const LanguageSettings = () => {
  const {
    settings: { theme },
  } = useSettingsStore();
  return (
    <View style={{ marginLeft: 10 }}>
      <DropdownSelect
        placeholder="Change Theme."
        options={languages}
        optionLabel={"name"}
        optionValue={"code"}
        selectedValue={languages[0].code}
        isMultiple={false}
        helperText="You can configure your default app language here."
        dropdownContainerStyle={{
          marginBottom: 0,
          maxWidth: 300,
        }}
        selectedItemStyle={{
          color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          fontFamily: FONTS.regular,
        }}
        dropdownIconStyle={{ top: 15, right: 15 }}
        dropdownStyle={{
          borderWidth: 0,
          paddingVertical: 8,
          paddingHorizontal: 20,
          minHeight: 40,
          maxWidth: 300,
          backgroundColor:
            theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
        }}
        placeholderStyle={{
          fontFamily: FONTS.regular,
          fontSize: 18,
        }}
        onValueChange={(value: string) => console.log({ value })}
        labelStyle={{ fontFamily: FONTS.regularBold, fontSize: 20 }}
        primaryColor={
          theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary
        }
        dropdownHelperTextStyle={{
          color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          fontFamily: FONTS.regular,
          fontSize: 15,
        }}
        modalOptionsContainerStyle={{
          padding: 10,
          backgroundColor:
            theme === "dark" ? COLORS.dark.main : COLORS.light.main,
        }}
        checkboxComponentStyles={{
          checkboxSize: 10,
          checkboxStyle: {
            backgroundColor:
              theme === "dark" ? COLORS.dark.secondary : COLORS.light.secondary,
            borderRadius: 10,
            padding: 5,
            borderColor:
              theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary,
          },
          checkboxLabelStyle: {
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
            fontSize: 18,
            fontFamily: FONTS.regular,
          },
        }}
      />
    </View>
  );
};

export default LanguageSettings;
