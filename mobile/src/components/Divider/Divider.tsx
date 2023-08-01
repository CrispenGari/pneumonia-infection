import { View, Text } from "react-native";
import React from "react";
import { styles } from "../../styles";

interface Props {
  title: string;
  color: string;
  centered: boolean;
}
const Divider: React.FunctionComponent<Props> = ({
  title,
  color,
  centered,
}) => {
  return (
    <View
      style={{
        marginVertical: 10,
        flexDirection: "row",
        width: "100%",
        justifyContent: "space-between",
        alignItems: "center",
        marginLeft: centered ? 0 : 10,
      }}
    >
      {centered ? (
        <View
          style={{
            borderBottomColor: color,
            flex: 1,
            borderBottomWidth: 0.5,
            marginRight: 10,
          }}
        />
      ) : null}

      <Text style={[styles.h1, { color }]}>{title}</Text>
      <View
        style={{
          borderBottomColor: color,
          flex: 1,
          borderBottomWidth: 0.5,
          marginLeft: 10,
        }}
      />
    </View>
  );
};

export default Divider;
