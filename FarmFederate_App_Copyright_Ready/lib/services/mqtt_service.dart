typedef SensorCallback = void Function(Map<String, dynamic> sensors);

class MqttService {
  final String host;
  final SensorCallback? onSensor;
  MqttService({required this.host, this.onSensor});

  Future<void> connect() async {
    return;
  }

  void disconnect() {}

  Future<void> publish(String topic, String payload) async {}
}
