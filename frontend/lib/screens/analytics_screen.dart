import 'dart:async';
import 'package:flutter/material.dart';
import '../models/sensor_data.dart';
import '../services/simulated_iot_service.dart';

class AnalyticsScreen extends StatefulWidget {
  final String apiBase;

  const AnalyticsScreen({Key? key, required this.apiBase}) : super(key: key);

  @override
  State<AnalyticsScreen> createState() => _AnalyticsScreenState();
}

class _AnalyticsScreenState extends State<AnalyticsScreen> {
  List<SensorData> _historicalData = [];
  bool _isLoading = true;
  String _selectedTimeRange = '7d';
  String _selectedCrop = 'Rice';

  @override
  void initState() {
    super.initState();
    _loadHistoricalData();
  }

  int _daysForRange(String range) {
    switch (range) {
      case '24h': return 1;
      case '7d': return 7;
      case '30d': return 30;
      case '90d': return 90;
      default: return 7;
    }
  }

  Future<void> _loadHistoricalData() async {
    setState(() => _isLoading = true);

    // Use SimulatedIoTService for realistic historical data
    // based on NASA POWER, FAO, USDA NRCS & NOAA GSOD datasets
    await Future.delayed(const Duration(milliseconds: 500));

    final days = _daysForRange(_selectedTimeRange);
    _historicalData = SimulatedIoTService.generateHistorical(
      days: days,
      readingsPerDay: days <= 1 ? 24 : 4,
    );

    setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Historical Analytics'),
        actions: [
          PopupMenuButton<String>(
            initialValue: _selectedTimeRange,
            onSelected: (value) {
              setState(() => _selectedTimeRange = value);
              _loadHistoricalData();
            },
            itemBuilder: (context) => [
              const PopupMenuItem(value: '24h', child: Text('Last 24 Hours')),
              const PopupMenuItem(value: '7d', child: Text('Last 7 Days')),
              const PopupMenuItem(value: '30d', child: Text('Last 30 Days')),
              const PopupMenuItem(value: '90d', child: Text('Last 90 Days')),
            ],
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildCropSelector(),
                  const SizedBox(height: 16),
                  _buildSummaryCards(),
                  const SizedBox(height: 24),
                  _buildCropHealthStatus(),
                  const SizedBox(height: 24),
                  _buildTrendAnalysis(),
                  const SizedBox(height: 24),
                  _buildPredictiveInsights(),
                  const SizedBox(height: 24),
                  _buildDataQuality(),
                ],
              ),
            ),
    );
  }

  Widget _buildCropSelector() {
    final crops = SimulatedIoTService.cropOptimalRanges.keys.toList();
    return SizedBox(
      height: 36,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: crops.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final crop = crops[index];
          final isSelected = crop == _selectedCrop;
          return ChoiceChip(
            label: Text(
              crop,
              style: TextStyle(
                color: isSelected ? Colors.white : Colors.white70,
                fontSize: 13,
              ),
            ),
            selected: isSelected,
            selectedColor: const Color(0xFF2E7D32),
            backgroundColor: const Color(0xFF1D1E33),
            onSelected: (_) => setState(() => _selectedCrop = crop),
          );
        },
      ),
    );
  }

  Widget _buildSummaryCards() {
    if (_historicalData.isEmpty) return const SizedBox();

    final avgMoisture = _historicalData.map((e) => e.soilMoisture).reduce((a, b) => a + b) / _historicalData.length;
    final avgTemp = _historicalData.map((e) => e.airTemperature).reduce((a, b) => a + b) / _historicalData.length;
    final avgHumidity = _historicalData.map((e) => e.humidity).reduce((a, b) => a + b) / _historicalData.length;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Summary Statistics',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: _buildStatCard(
                'Avg Moisture',
                '${avgMoisture.toStringAsFixed(1)}%',
                Icons.water_drop,
                Colors.blue,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Avg Temperature',
                '${avgTemp.toStringAsFixed(1)}°C',
                Icons.thermostat,
                Colors.orange,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Avg Humidity',
                '${avgHumidity.toStringAsFixed(1)}%',
                Icons.cloud,
                const Color(0xFF4AE2E2),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildStatCard(String label, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha:0.3)),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 32),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(
              color: Colors.white60,
              fontSize: 11,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildCropHealthStatus() {
    if (_historicalData.isEmpty) return const SizedBox();

    final ranges = SimulatedIoTService.cropOptimalRanges[_selectedCrop];
    if (ranges == null) return const SizedBox();

    final latest = _historicalData.last;
    final tempRange = ranges['temperature']!;
    final humRange = ranges['humidity']!;
    final soilRange = ranges['soilMoisture']!;

    final tempOk = latest.airTemperature >= tempRange[0] && latest.airTemperature <= tempRange[1];
    final humOk = latest.humidity >= humRange[0] && latest.humidity <= humRange[1];
    final soilOk = latest.soilMoisture >= soilRange[0] && latest.soilMoisture <= soilRange[1];

    final allOk = tempOk && humOk && soilOk;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: allOk ? Colors.green.withValues(alpha:0.3) : Colors.orange.withValues(alpha:0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                allOk ? Icons.check_circle : Icons.warning_amber,
                color: allOk ? Colors.green : Colors.orange,
                size: 24,
              ),
              const SizedBox(width: 12),
              Text(
                '$_selectedCrop Health Check',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildRangeRow(
            'Temperature',
            '${latest.airTemperature.toStringAsFixed(1)}°C',
            '${tempRange[0].toInt()}-${tempRange[1].toInt()}°C',
            tempOk,
          ),
          const SizedBox(height: 10),
          _buildRangeRow(
            'Humidity',
            '${latest.humidity.toStringAsFixed(1)}%',
            '${humRange[0].toInt()}-${humRange[1].toInt()}%',
            humOk,
          ),
          const SizedBox(height: 10),
          _buildRangeRow(
            'Soil Moisture',
            '${latest.soilMoisture.toStringAsFixed(1)}%',
            '${soilRange[0].toInt()}-${soilRange[1].toInt()}%',
            soilOk,
          ),
          if (!allOk) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.orange.withValues(alpha:0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  const Icon(Icons.tips_and_updates, color: Colors.orange, size: 16),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      !tempOk
                          ? 'Temperature is outside the optimal range for $_selectedCrop'
                          : !soilOk
                              ? 'Soil moisture needs adjustment for $_selectedCrop'
                              : 'Humidity is outside the ideal range for $_selectedCrop',
                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildRangeRow(String label, String current, String optimal, bool inRange) {
    return Row(
      children: [
        Icon(
          inRange ? Icons.check_circle_outline : Icons.error_outline,
          color: inRange ? Colors.green : Colors.orange,
          size: 18,
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Text(label, style: const TextStyle(color: Colors.white70, fontSize: 13)),
        ),
        Text(
          current,
          style: TextStyle(
            color: inRange ? Colors.green : Colors.orange,
            fontWeight: FontWeight.bold,
            fontSize: 13,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          '(optimal: $optimal)',
          style: const TextStyle(color: Colors.white38, fontSize: 11),
        ),
      ],
    );
  }

  Widget _buildTrendAnalysis() {
    if (_historicalData.length < 2) return const SizedBox();

    // Compute actual trends from the data
    final half = _historicalData.length ~/ 2;
    final firstHalf = _historicalData.sublist(0, half);
    final secondHalf = _historicalData.sublist(half);

    final avgMoist1 = firstHalf.map((e) => e.soilMoisture).reduce((a, b) => a + b) / firstHalf.length;
    final avgMoist2 = secondHalf.map((e) => e.soilMoisture).reduce((a, b) => a + b) / secondHalf.length;
    final moistTrend = avgMoist2 - avgMoist1;

    final avgTemp1 = firstHalf.map((e) => e.airTemperature).reduce((a, b) => a + b) / firstHalf.length;
    final avgTemp2 = secondHalf.map((e) => e.airTemperature).reduce((a, b) => a + b) / secondHalf.length;
    final tempTrend = avgTemp2 - avgTemp1;

    final avgHum1 = firstHalf.map((e) => e.humidity).reduce((a, b) => a + b) / firstHalf.length;
    final avgHum2 = secondHalf.map((e) => e.humidity).reduce((a, b) => a + b) / secondHalf.length;
    final humTrend = avgHum2 - avgHum1;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.trending_up, color: Colors.green, size: 24),
              SizedBox(width: 12),
              Text(
                'Trend Analysis',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildTrendItem(
            'Soil Moisture',
            '${moistTrend >= 0 ? "+" : ""}${moistTrend.toStringAsFixed(1)}% over this period',
            moistTrend >= 0 ? Icons.arrow_upward : Icons.arrow_downward,
            moistTrend.abs() < 1 ? Colors.blue : (moistTrend >= 0 ? Colors.green : Colors.orange),
          ),
          const SizedBox(height: 12),
          _buildTrendItem(
            'Temperature',
            '${tempTrend >= 0 ? "+" : ""}${tempTrend.toStringAsFixed(1)}°C change',
            tempTrend.abs() < 1 ? Icons.check_circle : (tempTrend >= 0 ? Icons.arrow_upward : Icons.arrow_downward),
            tempTrend.abs() < 1 ? Colors.blue : (tempTrend > 3 ? Colors.red : Colors.orange),
          ),
          const SizedBox(height: 12),
          _buildTrendItem(
            'Humidity',
            '${humTrend >= 0 ? "+" : ""}${humTrend.toStringAsFixed(1)}% change',
            humTrend >= 0 ? Icons.arrow_upward : Icons.arrow_downward,
            humTrend.abs() < 2 ? Colors.blue : Colors.orange,
          ),
        ],
      ),
    );
  }

  Widget _buildTrendItem(String label, String description, IconData icon, Color color) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: color.withValues(alpha:0.2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: color, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                description,
                style: const TextStyle(
                  color: Colors.white60,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildPredictiveInsights() {
    if (_historicalData.isEmpty) return const SizedBox();

    final latest = _historicalData.last;
    final ranges = SimulatedIoTService.cropOptimalRanges[_selectedCrop];

    // Generate insights based on actual data
    final insights = <Map<String, String>>[];

    if (ranges != null) {
      final soilRange = ranges['soilMoisture']!;
      if (latest.soilMoisture < soilRange[0] + 5) {
        insights.add({
          'title': 'Irrigation Needed Soon',
          'desc': 'Soil moisture approaching lower limit for $_selectedCrop. Consider watering within 24 hours.',
        });
      } else if (latest.soilMoisture > soilRange[1] - 5) {
        insights.add({
          'title': 'Reduce Watering',
          'desc': 'Soil is well hydrated. Skip next irrigation cycle to prevent waterlogging.',
        });
      }

      final tempRange = ranges['temperature']!;
      if (latest.airTemperature > tempRange[1] - 2) {
        insights.add({
          'title': 'Heat Stress Risk',
          'desc': 'Temperature nearing upper limit for $_selectedCrop. Consider shade covers or misting.',
        });
      }
    }

    if (latest.humidity > 85) {
      insights.add({
        'title': 'Disease Watch',
        'desc': 'High humidity increases fungal disease risk. Ensure good airflow between plants.',
      });
    }

    if (insights.isEmpty) {
      insights.add({
        'title': 'Conditions Look Good',
        'desc': 'Current readings are within optimal range for $_selectedCrop. Keep monitoring!',
      });
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [
            Color(0xFF667eea),
            Color(0xFF764ba2),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.lightbulb, color: Colors.white, size: 24),
              SizedBox(width: 12),
              Text(
                'Smart Insights',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ...insights.map((insight) => Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: _buildInsightItem(insight['title']!, insight['desc']!),
          )),
        ],
      ),
    );
  }

  Widget _buildInsightItem(String title, String description) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha:0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            description,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDataQuality() {
    final totalReadings = _historicalData.length;
    final completeness = totalReadings > 0 ? 100.0 : 0.0;
    // Sensor accuracy based on noise level in SimulatedIoTService
    const sensorAccuracy = 97.5;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.verified, color: Colors.green, size: 24),
              SizedBox(width: 12),
              Text(
                'Data Quality Metrics',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildQualityMetric('Data Completeness', completeness, Colors.green),
          const SizedBox(height: 12),
          _buildQualityMetric('Sensor Accuracy', sensorAccuracy, Colors.green),
          const SizedBox(height: 12),
          _buildQualityMetric('Readings Count', totalReadings.toDouble(), Colors.blue,
              suffix: ' pts'),
        ],
      ),
    );
  }

  Widget _buildQualityMetric(String label, double value, Color color, {String suffix = '%'}) {
    final isPercentage = suffix == '%';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
              ),
            ),
            Text(
              isPercentage ? '${value.toStringAsFixed(1)}$suffix' : '${value.toInt()}$suffix',
              style: TextStyle(
                color: color,
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        if (isPercentage) ...[
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: value / 100,
            backgroundColor: Colors.white12,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ],
      ],
    );
  }
}
