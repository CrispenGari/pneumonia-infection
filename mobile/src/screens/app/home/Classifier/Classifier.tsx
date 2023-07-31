import { ScrollView, TouchableOpacity, View, Text } from "react-native";
import React from "react";
import { COLORS, KEYS, models, serverBaseURL } from "../../../../constants";
import { useSettingsStore } from "../../../../store";
import Form from "../../../../components/Form/Form";
import { generateRNFile, onImpact, retrieve, store } from "../../../../utils";
import ClassifierDate from "../../../../components/ClassifierDate/ClassifierDate";
import FormImage from "../../../../components/FormImage/FormImage";
import { styles } from "../../../../styles";
import RippleLoadingIndicator from "../../../../components/RippleLoadingIndicator/RippleLoadingIndicator";
import { useMutation } from "react-query";
import { ReactNativeFile } from "apollo-upload-client";
import { PredictionResponse } from "../../../../types";
import { HomeTabStacksNavProps } from "../../../../params";
import { MaterialIcons } from "@expo/vector-icons";

const Classifier: React.FunctionComponent<
  HomeTabStacksNavProps<"Classifier">
> = ({ navigation }) => {
  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerTitle: "Home",
      headerRight: () => (
        <TouchableOpacity
          style={{ marginHorizontal: 20 }}
          onPress={() => {
            if (settings.haptics) {
              onImpact();
            }
            navigation.navigate("History");
          }}
          activeOpacity={0.7}
        >
          <MaterialIcons
            name="history"
            size={24}
            color={theme === "dark" ? COLORS.common.white : COLORS.common.black}
          />
        </TouchableOpacity>
      ),
    });
  }, [navigation]);
  const [model, setModel] = React.useState<string>(models[0].version);
  const {
    settings: { theme, ...settings },
  } = useSettingsStore();
  const [image, setImage] = React.useState<{
    uri: string;
    name: string;
  } | null>(null);
  const { isLoading: diagnosing, mutateAsync } = useMutation({
    mutationKey: ["pneumonia", model],
    mutationFn: async (variables: {
      image: ReactNativeFile | null;
      model: string;
    }) => {
      const formData = new FormData();
      formData.append("image", variables.image as any);
      const res = await fetch(
        `${serverBaseURL}/api/${variables.model}/pneumonia`,
        {
          method: "POST",
          body: formData,
        }
      );
      const data = await res.json();
      return data as PredictionResponse;
    },
  });
  const diagnose = () => {
    if (!image) return;
    mutateAsync(
      {
        image: generateRNFile({ name: image.name, uri: image.uri }),
        model,
      },
      {
        onSuccess: async (data, _variables, _context) => {
          const hist = {
            date: new Date(),
            result: data,
          };
          const _histories = await retrieve(KEYS.HISTORY);
          const _hist = _histories ? JSON.parse(_histories) : [];
          const histories = [hist, _hist];
          await store(KEYS.HISTORY, JSON.stringify(histories));
          navigation.navigate("Results", {
            results: JSON.stringify(data),
            image: image.uri,
          });
        },
        onError(error, _variables, _context) {
          console.log({ error });
        },
      }
    );
  };

  return (
    <ScrollView
      scrollEventThrottle={16}
      showsHorizontalScrollIndicator={false}
      showsVerticalScrollIndicator={false}
      style={{
        flex: 1,
        backgroundColor:
          theme === "dark" ? COLORS.dark.main : COLORS.light.main,
      }}
      contentContainerStyle={{ padding: 10, paddingBottom: 140 }}
    >
      <ClassifierDate />
      <FormImage diagnosing={diagnosing} image={image} setImage={setImage} />
      <Form
        model={model}
        setModel={setModel}
        setImage={setImage}
        diagnosing={diagnosing}
        image={image}
      />
      <View
        style={{
          marginVertical: 10,
          alignSelf: "center",
          width: "100%",
          maxWidth: 400,
        }}
      >
        <TouchableOpacity
          disabled={diagnosing}
          onPress={() => {
            if (settings.haptics) {
              onImpact();
            }
            diagnose();
          }}
          activeOpacity={0.7}
          style={[
            styles.button,
            {
              maxWidth: 400,
              flexDirection: "row",
              backgroundColor:
                theme === "dark" ? COLORS.dark.tertiary : COLORS.dark.tertiary,
            },
          ]}
        >
          <Text
            style={[
              styles.button__text,
              {
                marginRight: diagnosing ? 10 : 0,
                color:
                  theme === "dark" ? COLORS.common.black : COLORS.common.black,
              },
            ]}
          >
            {diagnosing ? "DIAGNOSING" : "DIAGNOSE"}
          </Text>
          {diagnosing ? (
            <RippleLoadingIndicator
              color={theme === "dark" ? COLORS.dark.main : COLORS.dark.main}
              size={15}
            />
          ) : null}
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

export default Classifier;
