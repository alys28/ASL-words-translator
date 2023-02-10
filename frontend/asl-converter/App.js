import { StatusBar } from "expo-status-bar";
import { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  Pressable,
  ScrollView,
  AppRegistry,
} from "react-native";
import { Icon } from "react-native-elements";
import PhoneCamera from "./Components/PhoneCamera";
import { Camera, CameraType, requestCameraPermissionsAsync } from "expo-camera";

export default function App() {
  const [text, setText] = useState("");
  const [cameraView, setCameraView] = useState(false);
  const __startCamera = () => {
    const { status } = requestCameraPermissionsAsync();
    if (status === "granted") {
      setCameraView(true);
    } else {
      // Alert.alert("Access denied");
    }
  };
  const onPress = () => {
    if (text === "") {
      setTimeout(() => {
        setText("OK");
      }, 3000);
    } else {
      setText("");
    }
  };
  if (!cameraView) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>DeafAID👂</Text>
        <ScrollView style={styles.textSpace}>
          <Text style={{ color: "white", padding: 20, overflowY: "scroll" }}>
            {text ? text : "Press Mic Button and Speak..."}
          </Text>
        </ScrollView>
        <View>
          <Pressable
            onPress={() => {
              onPress();
            }}
            style={({ pressed }) => [
              {
                backgroundColor: pressed ? "rgb(210, 230, 255)" : "white",
              },
              styles.wrapperCustom,
            ]}
          >
            <Icon name="mic" reverse="true" />
          </Pressable>
          <Pressable
            onPress={() => {
              __startCamera();
            }}
            style={({ pressed }) => [
              {
                backgroundColor: pressed ? "rgb(210, 230, 255)" : "white",
              },
              styles.wrapperCustom,
            ]}
          >
            <Icon name="camera" reverse="true" />
          </Pressable>
        </View>
        <StatusBar style="auto" />
      </View>
    );
  } else {
    return <PhoneCamera />;
  }
}

AppRegistry.registerComponent(appName, () => App);

const styles = StyleSheet.create({
  container: {
    paddingTop: "10%",
    paddingBottom: "5%",
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "space-around",
  },
  title: {
    fontSize: "60px",
    fontWeight: "bold",
  },
  wrapperCustom: {},
  textSpace: {
    shadow: 2,
    backgroundColor: "#121212",
    borderRadius: 10,
    minHeight: 200,
    maxHeight: 250,
    width: "80%",
  },
});
