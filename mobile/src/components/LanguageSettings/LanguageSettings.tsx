import { View } from "react-native";
import React from "react";
import DropdownSelect from "react-native-input-select";
import { COLORS, FONTS, languages } from "../../constants";
import { useSettingsStore } from "../../store";

const LanguageSettings = () => {
  const [language, setLanguage] = React.useState("dark");

  const {
    settings: { theme },
  } = useSettingsStore();
  return (
    <View style={{ marginLeft: 10 }}>
      <DropdownSelect
        placeholder="Change Theme."
        options={languages}
        optionLabel={"name"}
        optionValue={"value"}
        selectedValue={language}
        isMultiple={false}
        helperText="You can configure your default app language here."
        dropdownContainerStyle={{
          marginBottom: 0,
          maxWidth: 300,
        }}
        dropdownStyle={{
          borderWidth: 0,
          padding: 0,
          margin: 0,
          minHeight: 30,
          maxWidth: 300,
          backgroundColor:
            theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary,
        }}
        placeholderStyle={{ fontFamily: FONTS.regularBold, fontSize: 20 }}
        onValueChange={(value: string) => setLanguage(value)}
        labelStyle={{ fontFamily: FONTS.regularBold, fontSize: 20 }}
        primaryColor={
          theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary
        }
        dropdownHelperTextStyle={{
          color: COLORS.common.black,
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
