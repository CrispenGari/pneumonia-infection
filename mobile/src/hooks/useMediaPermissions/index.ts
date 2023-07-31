import * as ImagePicker from "expo-image-picker";
import * as Camera from "expo-camera";
import React from "react";

export const useMediaPermissions = () => {
  const [permisions, setPermisions] = React.useState({
    camera: false,
    library: false,
  });
  React.useEffect(() => {
    (async () => {
      const { granted } = await ImagePicker.getMediaLibraryPermissionsAsync();
      if (granted) {
        setPermisions((state) => ({
          ...state,
          library: granted,
        }));
      } else {
        const { granted: g } =
          await ImagePicker.requestMediaLibraryPermissionsAsync();
        setPermisions((state) => ({
          ...state,
          library: g,
        }));
      }
    })();
  }, []);

  React.useEffect(() => {
    (async () => {
      const { granted } = await Camera.requestCameraPermissionsAsync();
      setPermisions((state) => ({
        ...state,
        camera: granted,
      }));
    })();
  }, []);
  return permisions;
};
