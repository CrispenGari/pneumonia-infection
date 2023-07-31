import { View, Text, TouchableOpacity, SafeAreaView } from "react-native";
import React from "react";
import { APP_NAME, COLORS, KEYS, logo } from "../../../constants";
import { AuthNavProps } from "../../../params";
import { styles } from "../../../styles";
import * as Animatable from "react-native-animatable";
import TypeWriter from "react-native-typewriter";
import { LinearGradient } from "expo-linear-gradient";
import { useSettingsStore } from "../../../store";
import { onImpact, store } from "../../../utils";
import { SettingsType } from "../../../types";

const messages = [
  "Hello welcome to our AI tool, pneumonia diagnosis made easier.",
  "Before using this tool, we recommend you to read the Terms and Conditions and the Privacy Policy of this mobile tool.",
  "Scan your chest X-Rays and check your pneumonia STATUS.",
  "Someone can have pneumonia that is caused by BACTERIA or VIRUS.",
  "Use our AI mobile tool to 80% accurately predict pneumonia from your chest X-ray images.",
];
const Landing: React.FunctionComponent<AuthNavProps<"Landing">> = ({
  navigation,
}) => {
  const {
    settings: { theme, ...settings },
    setSettings,
  } = useSettingsStore();
  React.useLayoutEffect(() => {
    navigation.setOptions({ headerShown: false });
  }, [navigation]);
  const [index, setIndex] = React.useState(0);
  React.useEffect(() => {
    const intervalId = setInterval(() => {
      if (index >= messages.length - 1) {
        setIndex(0);
      } else {
        setIndex((state) => state + 1);
      }
    }, 5000);
    return () => {
      clearInterval(intervalId);
    };
  }, [index]);

  return (
    <LinearGradient
      colors={
        theme === "dark"
          ? [COLORS.dark.tertiary, COLORS.dark.main]
          : [COLORS.light.tertiary, COLORS.light.main]
      }
      start={{
        x: 0,
        y: 1,
      }}
      end={{
        x: 0,
        y: 0,
      }}
      style={{ flex: 1, alignItems: "center" }}
    >
      <View
        style={{
          flex: 0.7,
          justifyContent: "center",
          alignItems: "center",
          maxWidth: 500,
        }}
      >
        <Text
          style={[
            styles.h1,
            {
              fontSize: 25,
              letterSpacing: 1,
              marginBottom: 20,
              color:
                theme === "dark" ? COLORS.common.white : COLORS.common.black,
            },
          ]}
        >
          {APP_NAME}
        </Text>
        <Animatable.Image
          animation={"bounce"}
          duration={2000}
          iterationCount={1}
          easing={"linear"}
          direction={"normal"}
          useNativeDriver={false}
          source={logo}
          style={{
            width: 100,
            height: 100,
            marginVertical: 30,
            resizeMode: "contain",
          }}
        />
        <View
          style={{
            width: "100%",
            maxWidth: 300,
            marginVertical: 10,
            padding: 10,
            margin: 10,
          }}
        >
          <TypeWriter
            style={[
              styles.p,
              {
                textAlign: "center",
                height: 100,
                color:
                  theme === "dark" ? COLORS.common.white : COLORS.common.black,
              },
            ]}
            typing={1}
            maxDelay={-50}
          >
            {messages[index]}
          </TypeWriter>
        </View>
      </View>
      <View
        style={{
          flex: 0.3,
          justifyContent: "center",
          alignItems: "center",
          width: "100%",
          maxWidth: 500,
        }}
      >
        <TouchableOpacity
          activeOpacity={0.7}
          onPress={async () => {
            if (settings.haptics) {
              onImpact();
            }
            const s: SettingsType = {
              ...settings,
              new: false,
              theme,
            };

            await store(KEYS.APP_SETTINGS, JSON.stringify(s));
            setSettings(s);
          }}
          style={[
            styles.button,
            {
              maxWidth: 300,
              justifyContent: "center",
              alignItems: "center",
              backgroundColor:
                theme === "dark" ? COLORS.dark.primary : COLORS.light.primary,
            },
          ]}
        >
          <Text
            style={[
              styles.button__text,
              {
                color:
                  theme === "dark" ? COLORS.common.white : COLORS.common.black,
              },
            ]}
          >
            CONTINUE
          </Text>
        </TouchableOpacity>
      </View>
      <SafeAreaView>
        <View
          style={{
            padding: 10,
            flexDirection: "row",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Text style={[styles.p]}>
            By using our AI Tool you are automatically accepting
          </Text>
          <TouchableOpacity activeOpacity={0.7}>
            <Text style={[styles.p, { color: COLORS.common.url }]}>
              {" "}
              Terms and Conditions
            </Text>
          </TouchableOpacity>
          <Text style={[styles.p]}> and you are agreeing with our</Text>
          <TouchableOpacity activeOpacity={0.7}>
            <Text style={[styles.p, { color: COLORS.common.url }]}>
              {" "}
              Privacy Policy.
            </Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </LinearGradient>
  );
};

export default Landing;
