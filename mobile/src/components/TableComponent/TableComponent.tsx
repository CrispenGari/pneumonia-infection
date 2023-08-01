import { View, Text } from "react-native";
import React from "react";
import { Table, Row, Rows } from "react-native-table-component";
import { COLORS, FONTS } from "../../constants";
import { styles } from "../../styles";
import { useSettingsStore } from "../../store";

interface Props {
  tableHead: string[];
  title: string;
  tableData: string[][];
}
const TableComponent: React.FC<Props> = ({ tableHead, title, tableData }) => {
  const {
    settings: { theme },
  } = useSettingsStore();

  return (
    <View
      style={{
        flex: 1,
        width: "100%",
        maxWidth: 400,
        alignSelf: "center",
      }}
    >
      <Text
        style={[
          styles.h1,
          {
            fontSize: 20,
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
            letterSpacing: 1,
            marginBottom: 10,
          },
        ]}
      >
        {title}
      </Text>
      <Table
        borderStyle={{
          borderWidth: 1,
          borderColor:
            theme === "dark" ? COLORS.dark.tertiary : COLORS.light.tertiary,
          borderRadius: 5,
        }}
      >
        <Row
          data={tableHead}
          style={{
            height: 40,
            backgroundColor:
              theme === "dark" ? COLORS.dark.secondary : COLORS.light.secondary,
          }}
          textStyle={{
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
            fontFamily: FONTS.regularBold,
            textAlign: "center",
            fontSize: 18,
          }}
        />
        <Rows
          data={tableData}
          textStyle={{
            margin: 6,
            fontFamily: FONTS.regular,
            textAlign: "center",
            fontSize: 18,
            color: theme === "dark" ? COLORS.common.white : COLORS.common.black,
          }}
        />
      </Table>
    </View>
  );
};

export default TableComponent;
