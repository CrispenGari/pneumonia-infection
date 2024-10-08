import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Haptics from "expo-haptics";
import * as Updates from "expo-updates";
import { Audio } from "expo-av";
import { Alert } from "react-native";
import { ReactNativeFile } from "apollo-upload-client";
import * as mime from "react-native-mime-types";
import { APP_NAME } from "../constants";
import * as StoreReview from "expo-store-review";

let diagnosingSound: Audio.Sound | undefined;
let resultSound: Audio.Sound | undefined;

export const onImpact = () => Haptics.impactAsync();
export const onNotification = () => Haptics.notificationAsync();

export const playDiagnosingSound = async () => {
  const { sound: s, status } = await Audio.Sound.createAsync(
    require("../../assets/sounds/diagnosing.wav"),
    {
      shouldPlay: true,
      isLooping: true,
      isMuted: false,
    }
  );
  if (status.isLoaded) {
    diagnosingSound = s;
  }
  if (!!diagnosingSound) {
    await diagnosingSound.playAsync().catch((err) => console.log(err));
  }
};

export const playResultSound = async () => {
  const { sound: s, status } = await Audio.Sound.createAsync(
    require("../../assets/sounds/results.wav"),
    {
      shouldPlay: true,
      isLooping: false,
      isMuted: false,
    }
  );
  if (status.isLoaded) {
    resultSound = s;
  }
  if (!!resultSound) {
    await resultSound.playAsync().catch((err) => console.log(err));
  }
};
export const stopDiagnosingSound = async () => {
  if (!!diagnosingSound) {
    await diagnosingSound.pauseAsync();
  }
};
export const stopResultSound = async () => {
  if (!!resultSound) {
    await resultSound.pauseAsync();
  }
};

export const rateApp = async () => {
  const available = await StoreReview.isAvailableAsync();
  if (available) {
    const hasAction = await StoreReview.hasAction();
    if (hasAction) {
      await StoreReview.requestReview();
    }
  }
};

export const generateRNFile = ({
  uri,
  name,
}: {
  uri: string;
  name: string;
}) => {
  return uri
    ? new ReactNativeFile({
        uri,
        type: mime.lookup(uri) || "image",
        name,
      })
    : null;
};
export const store = async (key: string, value: string): Promise<boolean> => {
  try {
    await AsyncStorage.setItem(key, value);
    return true;
  } catch (error: any) {
    return false;
  }
};

export const del = async (key: string): Promise<boolean> => {
  try {
    await AsyncStorage.removeItem(key);
    return true;
  } catch (error: any) {
    return false;
  }
};

export const retrieve = async (key: string): Promise<string | null> => {
  try {
    const data = await AsyncStorage.getItem(key);
    return data;
  } catch (error: any) {
    return null;
  }
};

export const onFetchUpdateAsync = async () => {
  try {
    const update = await Updates.checkForUpdateAsync();
    if (update.isAvailable) {
      await Updates.fetchUpdateAsync();
      await Updates.reloadAsync();
    }
  } catch (error) {
    Alert.alert(
      APP_NAME,
      error as any,
      [{ text: "OK", style: "destructive" }],
      { cancelable: false }
    );
  }
};
