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
import { onImpact } from "../../utils";

interface Props {
  diagnosing: boolean;
  setImage: React.Dispatch<
    React.SetStateAction<{
      uri: string;
      name: string;
    } | null>
  >;
  setModel: React.Dispatch<React.SetStateAction<string>>;
  model: string;
  image: {
    uri: string;
    name: string;
  } | null;
}
const Form: React.FunctionComponent<Props> = ({
  setImage,
  diagnosing,
  setModel,
  model,
  image,
}) => {
  const {
    settings: { theme, ...settings },
  } = useSettingsStore();
  const { camera, library } = useMediaPermissions();
  const openCamera = async () => {
    if (settings.haptics) {
      onImpact();
    }
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
        name: assets[0].fileName || "none.jpg",
      });
    }
  };
  const selectImage = async () => {
    if (settings.haptics) {
      onImpact();
    }
    if (!library) {
      Alert.alert(
        APP_NAME,
        `${APP_NAME} does not have permission to access your photos.`,
        [
          {
            style: "default",
            text: "Allow Access to all Photos",
            onPress: async () => {
              if (settings.haptics) {
                onImpact();
              }
              await ImagePicker.requestMediaLibraryPermissionsAsync();
              return;
            },
          },
          {
            style: "destructive",
            text: "CANCEL",
            onPress: () => {
              if (settings.haptics) {
                onImpact();
              }
            },
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
        name: assets[0].fileName || "none.jpg",
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

      {image ? (
        <>
          <TypeWriter
            style={[
              styles.h1,
              {
                fontSize: 20,
                marginBottom: 10,
                marginTop: 20,
                color:
                  theme === "dark" ? COLORS.common.white : COLORS.common.black,
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
            dropdownIconStyle={{ top: 15, right: 15 }}
            dropdownStyle={{
              borderWidth: 0,
              paddingVertical: 8,
              paddingHorizontal: 20,
              minHeight: 40,
              backgroundColor:
                theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary,
            }}
            placeholderStyle={{ fontFamily: FONTS.regular, fontSize: 18 }}
            onValueChange={(value: any) => setModel(value)}
            labelStyle={{ fontFamily: FONTS.regularBold, fontSize: 20 }}
            primaryColor={
              theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary
            }
            dropdownHelperTextStyle={{
              color:
                theme === "dark" ? COLORS.common.white : COLORS.common.black,
              fontFamily: FONTS.regular,
              fontSize: 15,
            }}
            helperText="Model version can be (v0 or v1) for MLP and LeNET respectively."
            modalOptionsContainerStyle={{
              padding: 10,
              backgroundColor:
                theme === "dark" ? COLORS.dark.main : COLORS.light.main,
            }}
            selectedItemStyle={{
              color:
                theme === "dark" ? COLORS.common.black : COLORS.common.white,
              fontSize: 18,
              fontFamily: FONTS.regular,
            }}
            checkboxComponentStyles={{
              checkboxSize: 10,
              checkboxStyle: {
                backgroundColor:
                  theme === "dark"
                    ? COLORS.dark.secondary
                    : COLORS.light.secondary,
                borderRadius: 10,
                padding: 5,
                borderColor:
                  theme === "dark"
                    ? COLORS.dark.tertiary
                    : COLORS.light.tertiary,
              },

              checkboxLabelStyle: {
                color:
                  theme === "dark" ? COLORS.common.white : COLORS.common.black,
                fontSize: 18,
                fontFamily: FONTS.regular,
              },
            }}
          />
        </>
      ) : null}
    </View>
  );
};

export default Form;
