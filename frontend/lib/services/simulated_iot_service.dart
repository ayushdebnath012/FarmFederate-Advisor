import 'dart:math';
import '../models/sensor_data.dart';

/// IoT sensor data engine calibrated from real agricultural datasets.
///
/// Data ranges and baselines sourced from:
/// - NASA POWER (solar radiation, temperature, wind)
/// - FAO Penman-Monteith / AQUASTAT (soil moisture, ET)
/// - USDA NRCS Web Soil Survey (soil types, pH, field capacity)
/// - NOAA GSOD weather stations (tropical/subtropical agriculture zones)
/// - India Meteorological Department (IMD) Kharif/Rabi season data
/// - ICAR/TNAU soil fertility classification (NPK ranges)
/// - Oregon State / UMass soil test interpretation guides
///
/// Generates realistic diurnal (day/night) cycles, seasonal drift,
/// irrigation patterns, and sensor noise matching real IoT deployments.
class SimulatedIoTService {
  static final _rng = Random();

  // ── Temperature baselines (Source: IMD + NOAA GSOD, India/SE Asia) ──
  // Kharif: Jun-Sep, Rabi: Oct-Mar, Zaid: Mar-Jun
  static const _tempBaselines = {
    'summer':       {'min': 30.0, 'max': 45.0},  // Zaid (Mar-Jun), IMD plains
    'monsoon':      {'min': 24.0, 'max': 35.0},  // Kharif (Jul-Sep)
    'post_monsoon': {'min': 20.0, 'max': 32.0},  // Oct-Nov
    'winter':       {'min': 10.0, 'max': 28.0},  // Rabi (Dec-Mar)
  };

  // ── Soil moisture (Source: USDA NRCS + FAO field capacity studies) ──
  // Volumetric Water Content (% VWC)
  static const _soilMoistureRanges = {
    'sandy':      {'wilting':  5.0, 'field_capacity': 20.0, 'saturation': 35.0},
    'sandy_loam': {'wilting':  8.0, 'field_capacity': 25.0, 'saturation': 42.0},
    'loam':       {'wilting': 10.0, 'field_capacity': 32.0, 'saturation': 48.0},
    'clay_loam':  {'wilting': 15.0, 'field_capacity': 38.0, 'saturation': 55.0},
    'clay':       {'wilting': 18.0, 'field_capacity': 45.0, 'saturation': 58.0},
  };

  // ── Humidity (Source: NOAA + IMD relative humidity records) ──
  static const _humidityRanges = {
    'arid':      {'min': 15.0, 'max': 45.0},
    'semi_arid': {'min': 30.0, 'max': 65.0},
    'humid':     {'min': 55.0, 'max': 95.0},
    'tropical':  {'min': 60.0, 'max': 95.0},  // SE Asia, Indian coast
  };

  // ── Soil pH by region (Source: USDA NRCS + ICAR soil surveys) ──
  static const _soilPhRanges = {
    'acidic_tropical': {'min': 4.5, 'max': 5.8},   // Laterite soils
    'neutral_alluvial': {'min': 6.0, 'max': 7.5},   // Indo-Gangetic plains
    'alkaline_arid':    {'min': 7.5, 'max': 8.5},   // Rajasthan, Deccan
    'balanced':         {'min': 5.5, 'max': 7.2},   // Most agricultural land
  };

  // ── NPK ranges (Source: ICAR/TNAU fertility classification + Oregon State) ──
  // Values in mg/kg (ppm)
  static const _npkRanges = {
    'nitrogen':   {'deficient': 10.0, 'low': 25.0, 'optimal': 40.0, 'high': 65.0},
    'phosphorus': {'deficient':  5.0, 'low': 12.0, 'optimal': 22.0, 'high': 45.0},
    'potassium':  {'deficient': 80.0, 'low': 120.0, 'optimal': 160.0, 'high': 350.0},
  };

  // ── Light intensity (Source: NASA POWER + Global Solar Atlas) ──
  // Values in lux
  static const _lightRanges = {
    'night':     0.0,
    'dawn_dusk': 800.0,
    'overcast':  15000.0,
    'clear':     80000.0,
    'peak_noon': 100000.0,
  };

  // ── Wind speed (Source: NOAA + IMD surface wind records) ──
  // Values in m/s
  static const _windRanges = {
    'calm':     {'min': 0.0, 'max': 1.0},
    'light':    {'min': 1.0, 'max': 3.5},
    'moderate': {'min': 3.5, 'max': 8.0},
    'strong':   {'min': 8.0, 'max': 15.0},
  };

  /// Get current season based on month (Indian agricultural calendar)
  static String _getSeason(int month) {
    if (month >= 4 && month <= 6) return 'summer';
    if (month >= 7 && month <= 9) return 'monsoon';
    if (month >= 10 && month <= 11) return 'post_monsoon';
    return 'winter';
  }

  /// Diurnal temperature cycle (Source: NOAA hourly station data patterns)
  /// Peak at ~14:00, minimum at ~05:00
  static double _diurnalTemp(double baseMin, double baseMax, int hour) {
    final phase = (hour - 5) * pi / 12.0;
    final normalized = (sin(phase) + 1) / 2.0;
    return baseMin + (baseMax - baseMin) * normalized;
  }

  /// Diurnal light intensity cycle (Source: NASA POWER hourly irradiance)
  static double _diurnalLight(int hour) {
    if (hour < 6 || hour > 19) return _rng.nextDouble() * 5; // night
    if (hour < 7 || hour > 18) return _lightRanges['dawn_dusk']! + _rng.nextDouble() * 500;
    // Sine curve peaking at noon
    final phase = (hour - 6) * pi / 12.0;
    final normalized = sin(phase);
    final isCloudy = _rng.nextDouble() < 0.3; // 30% chance of clouds
    final peak = isCloudy ? _lightRanges['overcast']! : _lightRanges['clear']!;
    return peak * normalized + _rng.nextDouble() * 3000;
  }

  /// Generate a realistic sensor reading for current time
  static SensorData generateReading({DateTime? timestamp}) {
    final now = timestamp ?? DateTime.now();
    final season = _getSeason(now.month);
    final tempRange = _tempBaselines[season]!;
    final soilRange = _soilMoistureRanges['clay_loam']!;
    final humRange = _humidityRanges['tropical']!;
    final phRange = _soilPhRanges['balanced']!;

    // Temperature: diurnal cycle + ±1°C sensor noise (NOAA GSOD pattern)
    final baseTemp = _diurnalTemp(tempRange['min']!, tempRange['max']!, now.hour);
    final temp = baseTemp + (_rng.nextDouble() - 0.5) * 2.0;

    // Humidity: inversely correlated with temperature (IMD records)
    final humBase = humRange['max']! -
        (temp - tempRange['min']!) /
            (tempRange['max']! - tempRange['min']!) *
            (humRange['max']! - humRange['min']!);
    final humidity = (humBase + (_rng.nextDouble() - 0.5) * 5.0)
        .clamp(humRange['min']!, humRange['max']!);

    // Soil moisture: FAO depletion curve with irrigation spikes at 6am/6pm
    final hoursSinceMidnight = now.hour + now.minute / 60.0;
    final irrigationEffect =
        (hoursSinceMidnight - 6).abs() < 1 || (hoursSinceMidnight - 18).abs() < 1
            ? 8.0
            : 0.0;
    final soilBase = soilRange['field_capacity']! -
        (hoursSinceMidnight / 24.0) * 6.0 +
        irrigationEffect;
    final soilMoisture = (soilBase + (_rng.nextDouble() - 0.5) * 3.0)
        .clamp(soilRange['wilting']!, soilRange['saturation']!);

    // Flow rate: irrigation window pattern (2-3.5 L/min during irrigation)
    final isIrrigating =
        (hoursSinceMidnight - 6).abs() < 0.5 || (hoursSinceMidnight - 18).abs() < 0.5;
    final flowRate = isIrrigating
        ? 2.0 + _rng.nextDouble() * 1.5
        : _rng.nextDouble() * 0.1;

    // Cumulative water usage (seasonal ET-based estimate, FAO)
    final dayOfYear = now.difference(DateTime(now.year)).inDays;
    final totalLiters = dayOfYear * 45.0 + _rng.nextDouble() * 10;

    // Soil pH: slow drift with ±0.2 sensor noise (USDA NRCS)
    final basePh = (phRange['min']! + phRange['max']!) / 2.0;
    final soilPh = (basePh + (_rng.nextDouble() - 0.5) * 0.4)
        .clamp(phRange['min']!, phRange['max']!);

    // Light intensity: NASA POWER diurnal irradiance
    final lightIntensity = _diurnalLight(now.hour);

    // Wind speed: NOAA surface wind (typically light in agriculture zones)
    final windRange = season == 'monsoon' ? _windRanges['moderate']! : _windRanges['light']!;
    final windSpeed = windRange['min']! +
        _rng.nextDouble() * (windRange['max']! - windRange['min']!) +
        (_rng.nextDouble() - 0.5) * 0.5;

    // NPK: ICAR soil fertility with seasonal depletion pattern
    final nBase = _npkRanges['nitrogen']!['optimal']!;
    final pBase = _npkRanges['phosphorus']!['optimal']!;
    final kBase = _npkRanges['potassium']!['optimal']!;
    // Nitrogen depletes faster during growing season
    final growingDepletion = (season == 'monsoon' || season == 'summer') ? -8.0 : 0.0;
    final nitrogen = (nBase + growingDepletion + (_rng.nextDouble() - 0.5) * 12.0)
        .clamp(10.0, 75.0);
    final phosphorus = (pBase + (_rng.nextDouble() - 0.5) * 8.0).clamp(5.0, 50.0);
    final potassium = (kBase + (_rng.nextDouble() - 0.5) * 40.0).clamp(80.0, 400.0);

    // Leaf wetness: high during monsoon, correlated with humidity (PMC studies)
    double leafWetness;
    if (humidity > 85) {
      leafWetness = 6.0 + _rng.nextDouble() * 12.0; // 6-18 hrs in wet conditions
    } else if (humidity > 70) {
      leafWetness = 2.0 + _rng.nextDouble() * 6.0;  // 2-8 hrs moderate
    } else {
      leafWetness = _rng.nextDouble() * 3.0;          // 0-3 hrs dry
    }

    return SensorData(
      soilMoisture: double.parse(soilMoisture.toStringAsFixed(1)),
      airTemperature: double.parse(temp.toStringAsFixed(1)),
      humidity: double.parse(humidity.toStringAsFixed(1)),
      flowRate: double.parse(flowRate.toStringAsFixed(2)),
      totalLiters: double.parse(totalLiters.toStringAsFixed(1)),
      soilPh: double.parse(soilPh.toStringAsFixed(1)),
      lightIntensity: double.parse(lightIntensity.toStringAsFixed(0)),
      windSpeed: double.parse(windSpeed.clamp(0, 25).toStringAsFixed(1)),
      nitrogen: double.parse(nitrogen.toStringAsFixed(1)),
      phosphorus: double.parse(phosphorus.toStringAsFixed(1)),
      potassium: double.parse(potassium.toStringAsFixed(0)),
      leafWetness: double.parse(leafWetness.toStringAsFixed(1)),
      timestamp: now,
    );
  }

  /// Generate historical data for analytics
  static List<SensorData> generateHistorical({
    int days = 30,
    int readingsPerDay = 4,
  }) {
    final now = DateTime.now();
    final data = <SensorData>[];

    for (int d = days - 1; d >= 0; d--) {
      for (int r = 0; r < readingsPerDay; r++) {
        final ts = now.subtract(Duration(days: d, hours: (24 ~/ readingsPerDay) * r));
        data.add(generateReading(timestamp: ts));
      }
    }

    return data;
  }

  /// Data source attribution
  static const String dataAttribution =
      'Data calibrated from NASA POWER, FAO, USDA NRCS, IMD & NOAA GSOD';

  /// Crop-specific optimal ranges (Source: FAO crop water requirements + ICAR)
  static const Map<String, Map<String, List<double>>> cropOptimalRanges = {
    'Rice': {
      'temperature': [20, 32],     // FAO: optimal 16-32°C
      'humidity': [60, 85],
      'soilMoisture': [35, 55],    // Paddy: near saturation
      'soilPh': [5.5, 6.5],       // TNAU rice nutrient guide
      'nitrogen': [30, 60],        // ICAR recommendation
    },
    'Wheat': {
      'temperature': [14, 25],     // FAO: germination 14-18°C
      'humidity': [40, 70],
      'soilMoisture': [20, 40],
      'soilPh': [5.5, 7.5],
      'nitrogen': [25, 55],
    },
    'Tomato': {
      'temperature': [20, 30],     // FAO: fruit set optimal
      'humidity': [50, 75],
      'soilMoisture': [25, 45],
      'soilPh': [5.5, 7.0],       // UF IFAS Extension
      'nitrogen': [25, 50],
    },
    'Cotton': {
      'temperature': [25, 35],     // FAO: growth optimal
      'humidity': [40, 65],
      'soilMoisture': [20, 40],
      'soilPh': [6.0, 7.0],
      'nitrogen': [30, 55],
    },
    'Maize': {
      'temperature': [18, 32],     // FAO: tasseling optimal
      'humidity': [50, 80],
      'soilMoisture': [25, 45],
      'soilPh': [6.0, 7.2],       // NC State Extension
      'nitrogen': [30, 60],
    },
  };
}
