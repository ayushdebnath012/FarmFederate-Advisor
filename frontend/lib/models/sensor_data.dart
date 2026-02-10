// Models for sensor data
class SensorData {
  final double soilMoisture;
  final double airTemperature;
  final double humidity;
  final double flowRate;
  final double totalLiters;
  final double soilPh;
  final double lightIntensity; // lux
  final double windSpeed; // m/s
  final double nitrogen; // mg/kg (ppm)
  final double phosphorus; // mg/kg (ppm)
  final double potassium; // mg/kg (ppm)
  final double leafWetness; // hours/day
  final DateTime timestamp;

  SensorData({
    required this.soilMoisture,
    required this.airTemperature,
    required this.humidity,
    required this.flowRate,
    required this.totalLiters,
    this.soilPh = 6.5,
    this.lightIntensity = 50000,
    this.windSpeed = 2.0,
    this.nitrogen = 35.0,
    this.phosphorus = 20.0,
    this.potassium = 150.0,
    this.leafWetness = 3.0,
    required this.timestamp,
  });

  factory SensorData.fromJson(Map<String, dynamic> json) {
    return SensorData(
      soilMoisture: (json['soil_moisture'] ?? 0).toDouble(),
      airTemperature: (json['temperature'] ?? json['air_temperature'] ?? 0).toDouble(),
      humidity: (json['humidity'] ?? 0).toDouble(),
      flowRate: (json['flow_rate'] ?? 0).toDouble(),
      totalLiters: (json['total_liters'] ?? 0).toDouble(),
      soilPh: (json['soil_ph'] ?? 6.5).toDouble(),
      lightIntensity: (json['light_intensity'] ?? 50000).toDouble(),
      windSpeed: (json['wind_speed'] ?? 2.0).toDouble(),
      nitrogen: (json['nitrogen'] ?? 35.0).toDouble(),
      phosphorus: (json['phosphorus'] ?? 20.0).toDouble(),
      potassium: (json['potassium'] ?? 150.0).toDouble(),
      leafWetness: (json['leaf_wetness'] ?? 3.0).toDouble(),
      timestamp: DateTime.parse(json['timestamp'] ?? DateTime.now().toIso8601String()),
    );
  }
}

class ControlState {
  final bool waterPump;
  final bool relay;
  final bool solenoidValve;

  ControlState({
    this.waterPump = false,
    this.relay = false,
    this.solenoidValve = false,
  });

  ControlState copyWith({
    bool? waterPump,
    bool? relay,
    bool? solenoidValve,
  }) {
    return ControlState(
      waterPump: waterPump ?? this.waterPump,
      relay: relay ?? this.relay,
      solenoidValve: solenoidValve ?? this.solenoidValve,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'water_pump': waterPump,
      'relay': relay,
      'solenoid_valve': solenoidValve,
    };
  }
}
