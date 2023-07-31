import { View, TouchableOpacity, Alert } from "react-native";
import React from "react";
import { APP_NAME, COLORS, FONTS, models } from "../../constants";
import { useMediaPermissions } from "../../hooks";
import { useSettingsStore } from "../../store";
import * as ImagePicker from "expo-image-picker";
import * as Camera from "expo-camera";
import TypeWriter from "react-native-typewriter";
import { Ionicons, AntDesign } from "@expo/vector-icons";
import Dropdown from "react-native-input-select";
import { styles } from "../../styles";

interface Props {
  diagnosing: boolean;
  setImage: React.Dispatch<React.SetStateAction<any>>;
  setModel: React.Dispatch<React.SetStateAction<string>>;
  model: string;
}
const Form: React.FunctionComponent<Props> = ({
  setImage,
  diagnosing,
  setModel,
  model,
}) => {
  const {
    settings: { theme },
  } = useSettingsStore();
  const { camera, library } = useMediaPermissions();
  const openCamera = async () => {
    if (!camera) {
      Alert.alert(
        APP_NAME,
        `${APP_NAME} does not have permission to access your camera.`,
        [
          {
            style: "default",
            text: "Allow Permission",
            onPress: async () => {
              await Camera.requestCameraPermissionsAsync();
              return;
            },
          },
          {
            style: "destructive",
            text: "CANCEL",
            onPress: () => {},
          },
        ]
      );
      return;
    }
    const { assets, canceled } = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      base64: false,
      quality: 1,
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: false,
    });

    if (!canceled) {
      setImage({
        uri: assets[0].uri,
        name: assets[0].fileName,
      });
    }
  };
  const selectImage = async () => {
    if (!library) {
      Alert.alert(
        APP_NAME,
        `${APP_NAME} does not have permission to access your photos.`,
        [
          {
            style: "default",
            text: "Allow Access to all Photos",
            onPress: async () => {
              await ImagePicker.requestMediaLibraryPermissionsAsync();
              return;
            },
          },
          {
            style: "destructive",
            text: "CANCEL",
            onPress: () => {},
          },
        ]
      );
      return;
    }

    const { assets, canceled } = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      base64: false,
      quality: 1,
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: false,
    });

    if (!canceled) {
      setImage({
        uri: assets[0].uri,
        name: assets[0].fileName,
      });
    }
  };

  return (
    <View
      style={{
        alignSelf: "center",
        padding: 20,
        backgroundColor:
          theme === "dark" ? COLORS.dark.secondary : COLORS.light.secondary,
        borderRadius: 10,
        maxWidth: 400,
        marginVertical: 10,
        width: "100%",
      }}
    >
      <TypeWriter
        style={[
          styles.h1,
          {
            fontSize: 20,
            marginBottom: 20,
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          },
        ]}
        typing={1}
        maxDelay={-50}
      >
        Choose an image of a chest X-Ray or take a photo.
      </TypeWriter>
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-evenly",
        }}
      >
        <TouchableOpacity
          disabled={diagnosing}
          style={{
            backgroundColor:
              theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
            padding: 10,
            width: 50,
            height: 50,
            justifyContent: "center",
            alignItems: "center",
            borderRadius: 50,
          }}
          activeOpacity={0.7}
          onPress={openCamera}
        >
          <Ionicons
            name="camera"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        </TouchableOpacity>
        <TouchableOpacity
          disabled={diagnosing}
          style={{
            backgroundColor:
              theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
            padding: 10,
            width: 50,
            height: 50,
            justifyContent: "center",
            alignItems: "center",
            borderRadius: 50,
          }}
          activeOpacity={0.7}
          onPress={selectImage}
        >
          <AntDesign
            name="picture"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        </TouchableOpacity>
      </View>

      <TypeWriter
        style={[
          styles.h1,
          {
            fontSize: 20,
            marginBottom: 10,
            marginTop: 20,
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          },
        ]}
        typing={1}
        maxDelay={-50}
      >
        Select Model.
      </TypeWriter>

      <Dropdown
        placeholder="Select Model."
        options={models}
        optionLabel={"name"}
        optionValue={"version"}
        selectedValue={model}
        isMultiple={false}
        dropdownStyle={{
          borderWidth: 0,
          padding: 0,
          margin: 0,
          height: 10,
          backgroundColor:
            theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary,
        }}
        placeholderStyle={{ fontFamily: FONTS.regularBold, fontSize: 20 }}
        onValueChange={(value: any) => setModel(value)}
        labelStyle={{ fontFamily: FONTS.regularBold, fontSize: 20 }}
        primaryColor={
          theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary
        }
        dropdownHelperTextStyle={{
          color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          fontFamily: FONTS.regular,
          fontSize: 15,
        }}
        helperText="Model version can be (v0 or v1) for MLP and LeNET respectively."
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

export default Form;