import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../models/sensor_data.dart';
import '../services/auth_service.dart';
import 'ai_chat_screen.dart';
import 'analytics_screen.dart';
import 'federated_learning_screen.dart';
import 'multimodal_diagnosis_screen.dart';

class DashboardScreen extends StatefulWidget {
  final String apiBase;

  const DashboardScreen({Key? key, required this.apiBase}) : super(key: key);

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> with TickerProviderStateMixin {
  SensorData? _currentData;
  ControlState _controlState = ControlState();
  bool _isLoading = true;
  bool _notificationsEnabled = true;
  Timer? _refreshTimer;
  late AnimationController _pulseController;
  late TabController _tabController;
  final int _selectedIndex = 0;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 7, vsync: this);
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    _fetchSensorData();
    _refreshTimer = Timer.periodic(const Duration(seconds: 5), (_) => _fetchSensorData());
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _pulseController.dispose();
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _fetchSensorData() async {
    try {
      final response = await http.get(Uri.parse('${widget.apiBase}/sensors/latest'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _currentData = SensorData.fromJson(data);
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _sendControlCommand(String device, bool state) async {
    try {
      await http.post(
        Uri.parse('${widget.apiBase}/control/$device'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'state': state}),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to control $device: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFF1D1E33),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF1D976C), Color(0xFF93F9B9)],
                ),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.agriculture, size: 24),
            ),
            const SizedBox(width: 12),
            const Text(
              'FarmFederate Advisor',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 20,
                color: Color(0xFF93F9B9),
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: Icon(
              _notificationsEnabled ? Icons.notifications_active : Icons.notifications_off,
              color: _notificationsEnabled ? const Color(0xFF93F9B9) : Colors.white54,
            ),
            onPressed: () {
              setState(() {
                _notificationsEnabled = !_notificationsEnabled;
              });
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(
                    _notificationsEnabled
                        ? 'Alerts enabled'
                        : 'Alerts disabled',
                  ),
                  duration: const Duration(seconds: 2),
                ),
              );
            },
          ),
          PopupMenuButton(
            icon: const Icon(Icons.account_circle),
            itemBuilder: (context) => [
              const PopupMenuItem(value: 'profile', child: Text('Profile')),
              const PopupMenuItem(value: 'settings', child: Text('Settings')),
              const PopupMenuItem(value: 'logout', child: Text('Logout')),
            ],
            onSelected: (value) {
              if (value == 'logout') {
                AuthService().signOut();
                Navigator.pushReplacementNamed(context, '/login');
              } else if (value == 'profile') {
                Navigator.pushNamed(context, '/profile');
              } else if (value == 'settings') {
                Navigator.pushNamed(context, '/settings');
              }
            },
          ),
          // Login button for guests (far right)
          FutureBuilder(
            future: AuthService().currentUser != null ? Future.value(true) : Future.value(false),
            builder: (context, snapshot) {
              final isLoggedIn = snapshot.data ?? false;
              if (!isLoggedIn) {
                return TextButton.icon(
                  onPressed: () {
                    Navigator.pushReplacementNamed(context, '/login');
                  },
                  icon: const Icon(Icons.login, color: Colors.white),
                  label: const Text(
                    'Login',
                    style: TextStyle(color: Colors.white),
                  ),
                );
              }
              return const SizedBox.shrink();
            },
          ),
        ],
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: const Color(0xFF1D976C),
          indicatorWeight: 3,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.white70,
          labelStyle: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
          unselectedLabelStyle: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.normal,
          ),
          isScrollable: true,
          tabs: const [
            Tab(icon: Icon(Icons.dashboard), text: 'Overview'),
            Tab(icon: Icon(Icons.terrain), text: 'Soil Analytics'),
            Tab(icon: Icon(Icons.smart_toy), text: 'AI Assistant'),
            Tab(icon: Icon(Icons.local_florist), text: 'Crop Check'),
            Tab(icon: Icon(Icons.trending_up), text: 'Analytics'),
            Tab(icon: Icon(Icons.group), text: 'Farm Network'),
            Tab(icon: Icon(Icons.settings), text: 'Controls'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildOverviewTab(),
          _buildSoilAnalyticsTab(),
          _buildAIChatTab(),
          _buildCropCheckTab(),
          _buildHistoricalAnalyticsTab(),
          _buildFederatedLearningTab(),
          _buildControlsTab(),
        ],
      ),
    );
  }

  Widget _buildOverviewTab() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    return RefreshIndicator(
      onRefresh: _fetchSensorData,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildStatusHeader(),
            const SizedBox(height: 24),
            _buildAnalyticsCards(),
            const SizedBox(height: 24),
            _buildMQTTStatus(),
            const SizedBox(height: 24),
            const Text(
              'Live Monitoring',
              style: TextStyle(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            _buildSensorGrid(),
            const SizedBox(height: 24),
            _buildQuickActions(),
          ],
        ),
      ),
    );
  }

  Widget _buildSoilAnalyticsTab() {
    if (_isLoading || _currentData == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return RefreshIndicator(
      onRefresh: _fetchSensorData,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Soil Health Analysis',
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Real-time soil monitoring and recommendations',
              style: TextStyle(
                color: Colors.white.withOpacity(0.7),
                fontSize: 14,
              ),
            ),
            const SizedBox(height: 24),
            _buildSoilConditionsCard(),
            const SizedBox(height: 24),
            _buildSoilDetailedMetrics(),
            const SizedBox(height: 24),
            _buildSoilRecommendations(),
          ],
        ),
      ),
    );
  }

  Widget _buildAIChatTab() {
    return AIChatScreen(apiBase: widget.apiBase);
  }

  Widget _buildCropCheckTab() {
    return MultimodalDiagnosisScreen(apiBase: widget.apiBase);
  }

  Widget _buildHistoricalAnalyticsTab() {
    return AnalyticsScreen(apiBase: widget.apiBase);
  }

  Widget _buildFederatedLearningTab() {
    return FederatedLearningScreen(apiBase: widget.apiBase);
  }

  Widget _buildControlsTab() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Environmental Controls',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Manage your farm automation systems',
            style: TextStyle(
              color: Colors.white.withOpacity(0.7),
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 24),
          _buildControlsSection(),
          const SizedBox(height: 24),
          _buildScheduledControls(),
        ],
      ),
    );
  }

  Widget _buildAnalyticsCards() {
    if (_currentData == null) return const SizedBox();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Quick Analytics',
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: _buildAnalyticsCard(
                'Soil Health',
                _getOverallSoilCondition(),
                Icons.eco,
                _getOverallSoilCondition() == 'Optimal' ? Colors.green : Colors.orange,
                '${_calculateSoilHealthScore()}%',
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: _buildAnalyticsCard(
                'Water Status',
                '${_currentData!.soilMoisture.toStringAsFixed(0)}%',
                Icons.water_drop,
                _getMoistureColor(_currentData!.soilMoisture),
                _getMoistureStatus(_currentData!.soilMoisture),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: _buildAnalyticsCard(
                'Temperature',
                '${_currentData!.airTemperature.toStringAsFixed(1)}°C',
                Icons.thermostat,
                _getTemperatureColor(_currentData!.airTemperature),
                _getTemperatureStatus(_currentData!.airTemperature),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: _buildAnalyticsCard(
                'Alerts',
                _getActiveAlertsCount().toString(),
                Icons.warning_amber,
                _getActiveAlertsCount() > 0 ? Colors.red : Colors.green,
                _getActiveAlertsCount() > 0 ? 'Active' : 'All Clear',
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildAnalyticsCard(String title, String value, IconData icon, Color color, String subtitle) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(icon, color: color, size: 20),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  int _calculateSoilHealthScore() {
    if (_currentData == null) return 0;
    
    int score = 100;
    final moisture = _currentData!.soilMoisture;
    final temp = _currentData!.airTemperature;
    final humidity = _currentData!.humidity;
    
    // Deduct points for out-of-range values
    if (moisture < 40 || moisture > 60) score -= 25;
    if (moisture < 30 || moisture > 70) score -= 25;
    
    if (temp < 18 || temp > 28) score -= 20;
    if (temp < 15 || temp > 32) score -= 20;
    
    if (humidity < 40 || humidity > 80) score -= 10;
    
    return score.clamp(0, 100);
  }

  int _getActiveAlertsCount() {
    if (_currentData == null) return 0;
    
    int alerts = 0;
    if (_currentData!.soilMoisture < 30 || _currentData!.soilMoisture > 70) alerts++;
    if (_currentData!.airTemperature < 15 || _currentData!.airTemperature > 32) alerts++;
    if (_currentData!.humidity < 35 || _currentData!.humidity > 85) alerts++;
    
    return alerts;
  }

  Widget _buildSoilDetailedMetrics() {
    if (_currentData == null) return const SizedBox();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Detailed Metrics',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        _buildMetricRow('Soil Moisture', '${_currentData!.soilMoisture.toStringAsFixed(1)}%', 40, 60, _currentData!.soilMoisture),
        const SizedBox(height: 12),
        _buildMetricRow('Air Temperature', '${_currentData!.airTemperature.toStringAsFixed(1)}°C', 18, 28, _currentData!.airTemperature),
        const SizedBox(height: 12),
        _buildMetricRow('Humidity', '${_currentData!.humidity.toStringAsFixed(1)}%', 40, 80, _currentData!.humidity),
        const SizedBox(height: 12),
        _buildMetricRow('Water Flow', '${_currentData!.flowRate.toStringAsFixed(2)} L/min', 0, 5, _currentData!.flowRate),
      ],
    );
  }

  Widget _buildMetricRow(String label, String value, double minOptimal, double maxOptimal, double currentValue) {
    final isOptimal = currentValue >= minOptimal && currentValue <= maxOptimal;
    final progress = ((currentValue - minOptimal) / (maxOptimal - minOptimal)).clamp(0.0, 1.0);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: const TextStyle(color: Colors.white70, fontSize: 14),
              ),
              Text(
                value,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: progress,
            backgroundColor: Colors.white12,
            valueColor: AlwaysStoppedAnimation(
              isOptimal ? Colors.green : Colors.orange,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Optimal range: ${minOptimal.toStringAsFixed(1)} - ${maxOptimal.toStringAsFixed(1)}',
            style: const TextStyle(color: Colors.white38, fontSize: 11),
          ),
        ],
      ),
    );
  }

  Widget _buildSoilRecommendations() {
    if (_currentData == null) return const SizedBox();

    List<Map<String, dynamic>> recommendations = [];
    
    if (_currentData!.soilMoisture < 40) {
      recommendations.add({
        'title': 'Increase Irrigation',
        'description': 'Soil moisture is below optimal. Consider increasing watering frequency.',
        'icon': Icons.water_drop,
        'color': Colors.blue,
      });
    } else if (_currentData!.soilMoisture > 60) {
      recommendations.add({
        'title': 'Reduce Watering',
        'description': 'Soil moisture is too high. Reduce irrigation to prevent root rot.',
        'icon': Icons.warning,
        'color': Colors.orange,
      });
    }
    
    if (_currentData!.airTemperature < 18) {
      recommendations.add({
        'title': 'Temperature Alert',
        'description': 'Air temperature is low. Monitor crop health and consider protection.',
        'icon': Icons.thermostat,
        'color': Colors.red,
      });
    } else if (_currentData!.airTemperature > 30) {
      recommendations.add({
        'title': 'High Temperature',
        'description': 'Air temperature is high. Ensure adequate watering and ventilation.',
        'icon': Icons.thermostat,
        'color': Colors.orange,
      });
    }
    
    if (_currentData!.humidity < 40) {
      recommendations.add({
        'title': 'Low Humidity',
        'description': 'Humidity is low. Consider misting or increasing watering.',
        'icon': Icons.cloud,
        'color': Colors.blue,
      });
    } else if (_currentData!.humidity > 80) {
      recommendations.add({
        'title': 'High Humidity',
        'description': 'Humidity is high. Ensure good air circulation to prevent disease.',
        'icon': Icons.cloud,
        'color': Colors.orange,
      });
    }

    if (recommendations.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.green.withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.green.withOpacity(0.3)),
        ),
        child: const Row(
          children: [
            Icon(Icons.check_circle, color: Colors.green, size: 32),
            SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'All Systems Optimal',
                    style: TextStyle(
                      color: Colors.green,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 4),
                  Text(
                    'Your soil conditions are within optimal ranges. Continue regular monitoring.',
                    style: TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Recommendations',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        ...recommendations.map((rec) => Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xFF1D1E33),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: rec['color'].withOpacity(0.3)),
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: rec['color'].withOpacity(0.2),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(rec['icon'], color: rec['color'], size: 24),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        rec['title'],
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        rec['description'],
                        style: const TextStyle(
                          color: Colors.white60,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        )),
      ],
    );
  }

  Widget _buildScheduledControls() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Scheduled Tasks',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        _buildScheduleCard(
          'Morning Irrigation',
          '06:00 AM',
          'Water pump runs for 15 minutes',
          Icons.water_drop,
          Colors.blue,
          true,
        ),
        const SizedBox(height: 12),
        _buildScheduleCard(
          'Evening Irrigation',
          '06:00 PM',
          'Secondary watering cycle',
          Icons.water,
          const Color(0xFF1D976C),
          true,
        ),
      ],
    );
  }

  Widget _buildScheduleCard(String title, String time, String description, IconData icon, Color color, bool isActive) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isActive ? color.withOpacity(0.5) : Colors.white12,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
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
                  time,
                  style: TextStyle(
                    color: color,
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  description,
                  style: const TextStyle(
                    color: Colors.white60,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
          Switch(
            value: isActive,
            onChanged: (value) {},
            activeThumbColor: color,
          ),
        ],
      ),
    );
  }

  Widget _buildMQTTStatus() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.green.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.green.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.router, color: Colors.green, size: 20),
                  ),
                  const SizedBox(width: 12),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'IoT Network Status',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        'MQTT Broker Connected',
                        style: TextStyle(
                          color: Colors.white60,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.green.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: Colors.green),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.check_circle, color: Colors.green, size: 14),
                    SizedBox(width: 6),
                    Text(
                      'Online',
                      style: TextStyle(
                        color: Colors.green,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _buildIoTDeviceCard('Devices', 'N/A', Icons.devices, Colors.blue, true),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildIoTDeviceCard('Gateway', '1 gateway', Icons.router, Colors.green, true),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildIoTDeviceCard('Controllers', '4 relays', Icons.toggle_on, Colors.orange, true),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildIoTDeviceCard(String name, String count, IconData icon, Color color, bool isOnline) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.3),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 18),
              const SizedBox(width: 8),
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: isOnline ? Colors.green : Colors.red,
                  shape: BoxShape.circle,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            name,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            count,
            style: TextStyle(
              color: color,
              fontSize: 10,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusHeader() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF1D976C), Color(0xFF93F9B9)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF1D976C).withOpacity(0.3),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Farm Status',
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    FadeTransition(
                      opacity: _pulseController,
                      child: Container(
                        width: 12,
                        height: 12,
                        decoration: const BoxDecoration(
                          color: Colors.greenAccent,
                          shape: BoxShape.circle,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    const Text(
                      'All Systems Optimal',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.eco, color: Colors.white, size: 32),
          ),
        ],
      ),
    );
  }

  Widget _buildSoilConditionsCard() {
    if (_currentData == null) {
      return Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: const Color(0xFF1E2746),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.white.withOpacity(0.1)),
        ),
        child: const Center(
          child: Text(
            'Loading soil conditions...',
            style: TextStyle(color: Colors.white70),
          ),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [
            Color(0xFF2C5364),
            Color(0xFF203A43),
            Color(0xFF0F2027),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 15,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.brown.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(
                  Icons.terrain,
                  color: Colors.brown,
                  size: 28,
                ),
              ),
              const SizedBox(width: 16),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Soil Conditions',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      'Real-time soil analysis',
                      style: TextStyle(
                        color: Colors.white60,
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Expanded(
                child: _buildSoilMetric(
                  'Moisture',
                  '${_currentData!.soilMoisture.toStringAsFixed(1)}%',
                  _getMoistureStatus(_currentData!.soilMoisture),
                  _getMoistureColor(_currentData!.soilMoisture),
                  Icons.water_drop,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildSoilMetric(
                  'Condition',
                  _getOverallSoilCondition(),
                  _getOverallSoilCondition() == 'Optimal' ? 'Good' : 'Attention',
                  _getOverallSoilCondition() == 'Optimal' ? Colors.green : Colors.orange,
                  Icons.check_circle,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSoilMetric(String label, String value, String status, Color color, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.3),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 20),
              const SizedBox(width: 8),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              status,
              style: TextStyle(
                color: color,
                fontSize: 11,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _getOverallSoilCondition() {
    if (_currentData == null) return 'Unknown';
    
    final moisture = _currentData!.soilMoisture;
    final temp = _currentData!.airTemperature;
    
    // Check if parameters are in optimal range (simplified for available sensors)
    if (moisture >= 40 && moisture <= 60 &&
        temp >= 18 && temp <= 28) {
      return 'Optimal';
    } else if (moisture >= 30 && moisture <= 70 &&
               temp >= 15 && temp <= 32) {
      return 'Good';
    } else {
      return 'Needs Attention';
    }
  }

  Color _getMoistureColor(double moisture) {
    if (moisture < 30) return Colors.red;
    if (moisture > 70) return Colors.blue;
    return Colors.green;
  }

  Color _getTemperatureColor(double temp) {
    if (temp < 15 || temp > 30) return Colors.red;
    if (temp < 18 || temp > 25) return Colors.orange;
    return Colors.green;
  }

  Color _getPHColor(double ph) {
    if (ph < 5.5 || ph > 8.0) return Colors.red;
    if (ph < 6.0 || ph > 7.5) return Colors.orange;
    return Colors.green;
  }

  Widget _buildSensorGrid() {
    if (_currentData == null) {
      return const Center(child: Text('No sensor data available'));
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        GridView.count(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisCount: 2,
          mainAxisSpacing: 16,
          crossAxisSpacing: 16,
          childAspectRatio: 1.1,
          children: [
            _buildSensorCard(
              'Temperature',
              '${_currentData!.airTemperature.toStringAsFixed(1)}°C',
              Icons.thermostat,
              const Color(0xFFE24A4A),
              _getTemperatureStatus(_currentData!.airTemperature),
            ),
            _buildSensorCard(
              'Humidity',
              '${_currentData!.humidity.toStringAsFixed(1)}%',
              Icons.cloud,
              const Color(0xFF4AE2E2),
              _getHumidityStatus(_currentData!.humidity),
            ),
            _buildSensorCard(
              'Soil Moisture',
              '${_currentData!.soilMoisture.toStringAsFixed(1)}%',
              Icons.water_drop,
              const Color(0xFF4A90E2),
              _getMoistureStatus(_currentData!.soilMoisture),
            ),
            _buildSensorCard(
              'Water Flow',
              '${_currentData!.flowRate.toStringAsFixed(2)} L/min',
              Icons.water,
              const Color(0xFF1D976C),
              'Active',
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildSensorCard(String title, String value, IconData icon, Color color, String status) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(icon, color: color, size: 24),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: _getStatusColor(status).withOpacity(0.2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  status,
                  style: TextStyle(
                    color: _getStatusColor(status),
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                value,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                title,
                style: TextStyle(
                  color: Colors.white.withOpacity(0.6),
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildControlsSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Irrigation Controls',
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        _buildControlCard(
          'Water Pump',
          'Main irrigation pump control',
          Icons.water,
          const Color(0xFF4A90E2),
          _controlState.waterPump,
          (value) {
            setState(() => _controlState = _controlState.copyWith(waterPump: value));
            _sendControlCommand('water_pump', value);
          },
        ),
        const SizedBox(height: 12),
        _buildControlCard(
          'Relay/Solenoid',
          'Water valve control',
          Icons.toggle_on,
          const Color(0xFF1D976C),
          _controlState.relay,
          (value) {
            setState(() => _controlState = _controlState.copyWith(relay: value));
            _sendControlCommand('relay', value);
          },
        ),
        const SizedBox(height: 12),
        _buildControlCard(
          'Solenoid Valve',
          'Irrigation valve control',
          Icons.water_drop,
          const Color(0xFF2196F3),
          _controlState.solenoidValve,
          (value) {
            setState(() => _controlState = _controlState.copyWith(solenoidValve: value));
            _sendControlCommand('solenoid_valve', value);
          },
        ),
      ],
    );
  }

  Widget _buildControlCard(
    String title,
    String subtitle,
    IconData icon,
    Color color,
    bool isActive,
    Function(bool) onChanged,
  ) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isActive ? color.withOpacity(0.5) : Colors.transparent,
          width: 2,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: color, size: 28),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.6),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          Transform.scale(
            scale: 0.9,
            child: Switch(
              value: isActive,
              onChanged: onChanged,
              activeThumbColor: color,
              activeTrackColor: color.withOpacity(0.5),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickActions() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Quick Actions',
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: _buildActionButton(
                'Export Data',
                Icons.download,
                const Color(0xFF11998E),
                () => _showExportDialog(),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildActionButton(
                'Generate Report',
                Icons.assessment,
                const Color(0xFFE24ACD),
                () => _showReportDialog(),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _buildActionButton(
                'IoT Devices',
                Icons.devices,
                const Color(0xFF667EEA),
                () => _showIoTDevicesDialog(),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildActionButton(
                'Settings',
                Icons.tune,
                const Color(0xFFFF6B6B),
                () {},
              ),
            ),
          ],
        ),
      ],
    );
  }

  void _showExportDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Export Data', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildExportOption('CSV Format', Icons.table_chart),
            const SizedBox(height: 12),
            _buildExportOption('JSON Format', Icons.code),
            const SizedBox(height: 12),
            _buildExportOption('PDF Report', Icons.picture_as_pdf),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }

  Widget _buildExportOption(String label, IconData icon) {
    return InkWell(
      onTap: () {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Exporting $label...')),
        );
      },
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.3),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          children: [
            Icon(icon, color: Colors.white70),
            const SizedBox(width: 12),
            Text(
              label,
              style: const TextStyle(color: Colors.white),
            ),
          ],
        ),
      ),
    );
  }

  void _showReportDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Generate Report', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Select report type:',
              style: TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 12),
            _buildReportOption('Daily Summary', Icons.today),
            const SizedBox(height: 8),
            _buildReportOption('Weekly Analysis', Icons.date_range),
            const SizedBox(height: 8),
            _buildReportOption('Monthly Overview', Icons.calendar_month),
            const SizedBox(height: 8),
            _buildReportOption('Custom Range', Icons.date_range),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }

  Widget _buildReportOption(String label, IconData icon) {
    return InkWell(
      onTap: () {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Generating $label report...')),
        );
      },
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.3),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          children: [
            Icon(icon, color: Colors.white70),
            const SizedBox(width: 12),
            Text(
              label,
              style: const TextStyle(color: Colors.white),
            ),
          ],
        ),
      ),
    );
  }

  void _showIoTDevicesDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('IoT Device Management', style: TextStyle(color: Colors.white)),
        content: SizedBox(
          width: double.maxFinite,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildDeviceItem('Device-01', 'Sensor Hub', true, '98%'),
              const SizedBox(height: 8),
              _buildDeviceItem('Camera-01', 'Field Camera', true, '95%'),
              const SizedBox(height: 8),
              _buildDeviceItem('Device-02', 'Weather Station', true, '100%'),
              const SizedBox(height: 8),
              _buildDeviceItem('Gateway', 'Main Gateway', true, '99%'),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Adding new device...')),
              );
            },
            child: const Text('Add Device'),
          ),
        ],
      ),
    );
  }

  Widget _buildDeviceItem(String name, String type, bool isOnline, String battery) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.3),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: isOnline ? Colors.green.withOpacity(0.3) : Colors.red.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: isOnline ? Colors.green : Colors.red,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  type,
                  style: const TextStyle(
                    color: Colors.white60,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                isOnline ? 'Online' : 'Offline',
                style: TextStyle(
                  color: isOnline ? Colors.green : Colors.red,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                '🔋 $battery',
                style: const TextStyle(
                  color: Colors.white60,
                  fontSize: 10,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton(String label, IconData icon, Color color, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [color, color.withOpacity(0.7)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          children: [
            Icon(icon, color: Colors.white, size: 32),
            const SizedBox(height: 8),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _getMoistureStatus(double value) {
    if (value < 30) return 'Low';
    if (value > 70) return 'High';
    return 'Optimal';
  }

  String _getTemperatureStatus(double value) {
    if (value < 15) return 'Cold';
    if (value > 30) return 'Hot';
    return 'Good';
  }

  String _getPHStatus(double value) {
    if (value < 6.0) return 'Acidic';
    if (value > 7.5) return 'Alkaline';
    return 'Optimal';
  }

  String _getHumidityStatus(double value) {
    if (value < 40) return 'Dry';
    if (value > 80) return 'Humid';
    return 'Good';
  }

  String _getLightStatus(double value) {
    if (value < 5000) return 'Low';
    if (value > 50000) return 'High';
    return 'Good';
  }

  Color _getStatusColor(String status) {
    switch (status.toLowerCase()) {
      case 'optimal':
      case 'good':
        return Colors.greenAccent;
      case 'low':
      case 'high':
      case 'cold':
      case 'hot':
        return Colors.orangeAccent;
      case 'acidic':
      case 'alkaline':
      case 'dry':
      case 'humid':
        return Colors.yellowAccent;
      default:
        return Colors.grey;
    }
  }
}
