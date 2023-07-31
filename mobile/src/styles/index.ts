import { StyleSheet } from "react-native";
import { FONTS } from "../constants";

export const styles = StyleSheet.create({
  h1: {
    fontFamily: FONTS.regularBold,
  },
  p: {
    fontFamily: FONTS.regular,
    fontSize: 16,
  },
  button: {
    width: "100%",
    maxWidth: 300,
    borderRadius: 999,
    padding: 15,
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
  },
  button__text: {
    fontFamily: FONTS.regularBold,
    fontSize: 18,
  },
});
